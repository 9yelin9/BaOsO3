# pymag/mlmag.py : select important features to predict magnetic structure

import os
num_thread = 16
os.environ['OMP_NUM_THREADS'] = str(num_thread)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_thread)

import re
import sys
import h5py
import time
import ctypes
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy import interpolate

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from sklearn import tree

# classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier	
from sklearn.svm import SVC

# resampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour

# feature selector
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.feature_selection import SelectFromModel

class MLMag:
	def __init__(self, save='dU1.0_UF8.0', num_thread=16, random_state=12):
		self.clib = ctypes.cdll.LoadLibrary(os.path.dirname(__file__)+'/libmod.so')		
		self.clib.GenKDOS.restype = ctypes.POINTER(ctypes.c_double)

		#self.peak_max = 2
		self.bins_max = 128
		self.Nkb      = 128
		self.Nkh      = 6
		self.Nb       = 12

		self.hsp_point   = [0, 22, 61, 83, 105, 127]
		self.hsp_label   = ['G', 'Z', 'S', 'X', 'U', 'R']
		self.hsp_label_g = [r'$\Gamma$', 'Z', 'S', 'X', 'U', 'R']
		#self.hsp_point   = [0, 24, 48, 127]
		#self.hsp_label   = ['G', 'X', 'M', 'R']
		#self.hsp_label_g = [r'$\Gamma$', 'X', 'M', 'R']

		self.type_dict = {'A':1, 'C':2, 'G':3}
		self.type_dict_r = {'1':'A', '2':'C', '3':'G'}
		self.params = ['type', 'JU', 'N', 'U', 'm', 'gap']

		self.tol_dict = {'m':1e-1, 'gap':1e-1, 'U':8.0, 'UF':4.0}

		self.save = save
		self.path_save = 'data/%s/' % self.save
		os.makedirs(self.path_save, exist_ok=True)

		self.random_state = random_state
		self.mc_dict = {
			'rf':   RandomForestClassifier(random_state=self.random_state, n_jobs=num_thread),
			'xgb':  XGBClassifier(random_state=self.random_state, nthread=num_thread),
			'lgbm': LGBMClassifier(random_state=self.random_state, n_jobs=num_thread),
			'cat':  CatBoostClassifier(random_state=self.random_state, thread_count=num_thread, silent=True, allow_writing_files=False),
			'lr':   LogisticRegression(random_state=self.random_state, n_jobs=num_thread, solver='sag', max_iter=100000),
			'svm':  SVC(random_state=self.random_state, probability=True),
		}
		self.rsp_dict = {
			'rus': RandomUnderSampler(random_state=self.random_state),
			'nm':  NearMiss(version=1),
			'enn': EditedNearestNeighbours(),
			'cnn': CondensedNearestNeighbour(random_state=self.random_state),
		}
		score_func = chi2
		self.ft_dict = {
			'kb':  GenericUnivariateSelect(score_func, mode='k_best', param=5000),
			'fwe': GenericUnivariateSelect(score_func, mode='fwe',    param=0.1),
			'rf':  SelectFromModel(estimator=RandomForestClassifier(random_state=self.random_state)),
		}

	def FnDict(self, fn):
		fn_dict = {
			'type':  self.type_dict[re.search('[A-Z]\d_JU', fn).group()[0]],
			'JU':    float(re.sub('JU',    '', re.search('JU\d+[.]\d+',        fn).group())),
			'N':     float(re.sub('N',     '', re.search('N\d+[.]\d+',         fn).group())),
			'U':     float(re.sub('_U',    '', re.search('_U\d+[.]\d+',        fn).group())),
			'n':     float(re.sub('_n',    '', re.search('_n\d+[.]\d+',        fn).group())),
			'm':     float(re.sub('_m',    '', re.search('_m[-]?\d+[.]\d+',    fn).group())),
			'e':     float(re.sub('_e',    '', re.search('_e[-]?\d+[.]\d+',    fn).group())),
			'fermi': float(re.sub('fermi', '', re.search('fermi[-]?\d+[.]\d+', fn).group())),
			'gap':   float(re.sub('gap',   '', re.search('gap[-]?\d+[.]\d+',   fn).group())),
		}
		fn_dict['e'] = np.around(fn_dict['e'])

		return fn_dict

	def FnDictD(self, fn):
		fn_dict = {
			'type':   int(re.sub('AF', '', re.search('AF\d',        fn).group())),
			'JU':   float(re.sub('_J', '', re.search('_J\d+[.]\d+', fn).group())),
			'N':    float(re.sub('Hz', '', re.search('Hz\d+[.]\d+', fn).group())),
			'U':    float(re.sub('UF', '', re.search('UF\d+[.]\d+', fn).group())),
			'm':    float(re.sub('_D', '', re.search('_D\d+[.]\d+', fn).group())),
			'gap':  float(re.sub('th', '', re.search('\d+th',       fn).group())),
		}
		
		return fn_dict

	def GroundOnly(self, fn_list):
		params = ['type', 'JU', 'N', 'U', 'e']

		data = np.zeros(len(params))
		for fn in fn_list:
			fn_dict = self.FnDict(fn)
			data = np.vstack((data, [fn_dict[p] for p in params]))
		data = np.delete(data, 0, axis=0)

		df = pd.DataFrame(data, columns=params)
		df = df.sort_values(by=['JU', 'N', 'U', 'e'])
		df = df.drop_duplicates(subset=['JU', 'N', 'U', 'type'], keep='first')
		grd_idx = df.index.to_list()

		return grd_idx

	def GenEnergy(self, bins, emin=-8, emax=8):
		bins = int(bins)

		t0 = time.time()

		en = '%s/energy_bins%d.txt' % (self.path_save, bins)
		e = np.linspace(emin, emax, bins)
		np.savetxt(en, e, fmt='%22.16f')

		t1 = time.time()
		print('GenEnergy(%s) : %fs' % (en, t1-t0))
	
	def GenParams(self, dtype, strain, tol):
		path_dos = '%s/%s_%s_%s%.2f' % (self.path_save, dtype, strain, tol, self.tol_dict[tol])
		os.makedirs(path_dos, exist_ok=True)

		pn = '%s/params.txt' % path_dos
		ln = '%s/fnlist.txt' % path_dos

		if os.path.isfile(pn):
			print('%s, %s already exist!' % (pn, ln))
			sys.exit()

		t0 = time.time()

		if dtype == 'ddos':
			path_dmft = 'dmft_old'
			dir_list = ['%s/%s/lattice/vdx' % (path_dmft, d) for d in os.listdir(path_dmft)\
					if (re.search('oDir', d)\
					and os.path.isfile('%s/%s/mkresult.bat' % (path_dmft, d))
					and float(re.sub('AF', '', re.search('AF\d', d).group())) > 0
					and float(re.sub('UF', '', re.search('UF\d+[.]\d+', d).group())) > self.tol_dict['UF'])]

			fn_list = ['%s/%s' % (d, f) for d in dir_list for f in os.listdir(d) if re.search('kG.*ep0.10', f)]

			with open(pn, 'w') as f:
				for p in self.params: f.write('%16s' % p)
				f.write('\n')
				for fn in fn_list:
					for p in self.params:
						f.write('%16.10f' % self.FnDictD(fn)[p])
					f.write('\n')

			with open(ln, 'w') as f:
				f.write('%d\n' % len(fn_list))
				f.write('\n'.join(['%s' % fn for fn in fn_list]))
		else:
			path_output = 'output/%s' % self.save
			dir_list = ['%s/%s/band_Nk%d' % (path_output, d, self.Nkb) for d in os.listdir(path_output)\
					if (re.search(strain, d) and not re.search('F\d_JU', d))]

			fn_list = ['%s/%s' % (d, f) for d in dir_list for f in os.listdir(d)]
			fn_list = [fn_list[i] for i in self.GroundOnly(fn_list)]
			
			if   tol == 'U':   fn_list = [fn for fn in fn_list if self.FnDict(fn)[tol] <= self.tol_dict['U']]
			elif tol == 'gap': fn_list = [fn for fn in fn_list if(self.FnDict(fn)[tol] >  self.tol_dict['gap'] and (10*self.FnDict(fn)['N']) % 10 == 0)]
			else :             fn_list = [fn for fn in fn_list if self.FnDict(fn)[tol] >  self.tol_dict[tol]]

			with open(pn, 'w') as fp, open(ln, 'w') as fl:
				fl.write('%d\n' % len(fn_list))
				for p in self.params: fp.write('%16s' % p)
				fp.write('\n')
				for fn in fn_list:
					fl.write('%s\n' % fn)
					if tol == 'gap':
						for _ in range(50):
							for p in self.params:
								fp.write('%16.10f' % self.FnDict(fn)[p])
							fp.write('\n')
					else:
						for p in self.params:
							fp.write('%16.10f' % self.FnDict(fn)[p])
						fp.write('\n')

		t1 = time.time()
		print('GenParams(%s, %s) : %fs' % (pn, ln, t1-t0))

	def GenDOS(self, dtype, strain, tol, bins, ep, is_under0=0, is_new0=0, is_linbrd=0, is_ranbrd=0):
		bins  = int(bins)
		ep    = float(ep)
		
		with open('%s/energy_bins%d.txt' % (self.path_save, bins), 'r') as f: e = np.genfromtxt(f)

		is_under0 = int(is_under0)
		is_new0   = int(is_new0)
		is_linbrd = int(is_linbrd)
		is_ranbrd = int(is_ranbrd)

		option = ''
		if is_under0:
			option += 'u'
			ws = [1 if ei < 0 else -1 for ei in e]
		else:
			ws = [1 for _ in e]

		if is_new0:
			option += 'n'
			dfermi = 1e-3

		if is_linbrd:
			option += 'l'
			e_min  = np.min(e)
			ep_min = 0.1
			eps = [ep_min + abs(ei / e_min) * ep for ei in e]
		elif is_ranbrd:
			option += 'r'
			ep_min = 0.1
			ep_max = 0.5
			eps = np.random.uniform(low=ep_min, high=ep_max, size=len(e))
		else:
			eps = [ep for _ in e]

		if len(option): option = '_' + option

		path_dos = '%s/%s_%s_%s%.2f' % (self.path_save, dtype, strain, tol, self.tol_dict[tol])
		ln = '%s/fnlist.txt' % path_dos
		dn = '%s/bins%d_ep%.2f%s.h5' % (path_dos, bins, ep, option)

		t0 = time.time()

		dos = np.zeros(1)
				
		if dtype == 'ldos':
			dos = np.zeros(bins)

			with open(ln, 'r') as f:
				f.readline() # skip header
				for fn in f:
					data = np.genfromtxt(fn.replace('band_Nk%d' % self.Nkb, 'dos_ep%.2f' % ep).strip(), skip_header=1)
					data_e = data[:, 0]
					data_d = np.sum(data[:, 1:], axis=1)

					itp = interpolate.interp1d(data_e, data_d)
					dos = np.vstack((dos, itp(e)))
		elif dtype == 'kdos' or dtype == 'hdos':
			Nk, k_list = (self.Nkb, range(self.Nkb)) if dtype == 'kdos' else (self.Nkh, self.hsp_point)

			dos  = np.zeros(Nk * bins)
			dosi = np.zeros((Nk, bins))

			with open(ln, 'r') as f:
				f.readline()
				for fn in f:
					data = np.genfromtxt(fn.strip(), skip_header=1)

					if is_new0:
						dntop = np.max(data[:, int(self.FnDict(fn)['N'])-1]) - dfermi
						upbot = np.min(data[:, int(self.FnDict(fn)['N'])])   + dfermi
						#Nfermi = (upbot - dntop) // dfermi
						fermi_list = np.linspace(dntop, upbot, 50)

						for fermi in fermi_list:
							for i, n in enumerate(k_list):
								for j in range(bins):
									dosi[i, j] = 0
									for k in range(self.Nb):
										dosi[i, j] += (eps[j] / ((e[j] - data[n, k] + fermi)**2 + eps[j]**2)) * data[n, k+self.Nb] * ws[j]
							dos = np.vstack((dos, np.ravel(dosi) / np.pi))
					else:
						for i, n in enumerate(k_list):
							for j in range(bins):
								dosi[i, j] = 0
								for k in range(self.Nb):
									dosi[i, j] += (eps[j] / ((e[j] - data[n, k])**2 + eps[j]**2)) * data[n, k+self.Nb] * ws[j]
						dos = np.vstack((dos, np.ravel(dosi) / np.pi))
		elif dtype == 'ddos':
			dos = np.zeros(self.Nkh * bins)

			with open(ln, 'r') as f:
				f.readline()
				for fn_G in f:
					fns = [re.sub('G', label, re.sub('ep0.02', 'ep%.2f' % ep, fn_G.strip())) for label in self.hsp_label]
					itps = []
					for fn in fns:
						with open(fn, 'r') as fp: data = np.genfromtxt(fp)
						data_e = data[:, 0]
						data_d = data[:, 1] * (self.Nb//2)

						itp = interpolate.interp1d(data_e, data_d, fill_value='extrapolate')
						itps.append(itp)
					dos = np.vstack((dos, np.ravel([itp(e) for itp in itps])))

		dos = np.delete(dos, 0, axis=0)

		with h5py.File(dn, 'w') as f: f.create_dataset('dos', data=dos, dtype='d')
		
		t1 = time.time()
		print('GenDOS(%s) : %fs' % (dn, t1-t0))

	def Preprocess(self, dn, mcn, ftn, rspn, feats_ft, is_eps):
		path_dos = dn.split('/')[0]
		dtype    = dn.split('_')[0]
		bins     = int(re.sub('bins', '', re.search('bins\d+', dn).group()))
		ep       = re.sub('ep', '', re.search('ep\d[.]\d+', dn).group())

		if   re.search('l', dtype): feats = self.params + ['x%d' % i for i in range(bins)] 
		elif re.search('k', dtype): feats = self.params + ['x%d_%d' % (j, i) for j in range(self.Nkb) for i in range(bins)]
		else:                       feats = self.params + ['%s%d' % (l, i) for l in self.hsp_label for i in range(bins)]

		pn = '%s/%s/params.txt' % (self.path_save, path_dos)
		with open(pn, 'r') as f: p = np.genfromtxt(f, skip_header=1)

		if is_eps:
			with h5py.File(self.path_save+re.sub('ep\d[.]\d+', 'ep%.2f'%multi_ep.min(), dn), 'r') as f: d = f['dos'][()]
			data = np.hstack((p, d))

			for ep in multi_ep:
				with h5py.File(self.path_save+re.sub('ep\d[.]\d+', 'ep%.2f'%ep, dn), 'r') as f: d = f['dos'][()]
				data = np.vstack((data, np.hstack((p, d))))
		else:
			with h5py.File(self.path_save+dn, 'r') as f: d = f['dos'][()]
			data = np.hstack((p, d))

		data = np.where(data < 0, 0, data)
		df = pd.DataFrame(data, columns=feats)
		df['type'] = df['type'].astype('int').astype('str').replace(self.type_dict_r)

		X = df.drop(self.params, axis=1)
		y = df['type']
		mc = self.mc_dict[mcn]

		if mcn == 'xgb':
			y = pd.get_dummies(y)

		ft = 0
		if re.search('gen_', ftn):
			ftn0 = re.sub('gen_', '', ftn)
			ft = self.ft_dict[ftn0]
			X = ft.fit_transform(np.array(X), y)
			fts = ft.get_feature_names_out(input_features=[ft for ft in feats if not ft in self.params])
			X = pd.DataFrame(X, columns=fts)

			fn = '%s/%s' % (self.path_save, re.sub('.h5', '_%s.txt' % ftn0, dn))
			np.savetxt(fn, fts, fmt='%s')
			print('%s generated.' % fn)
		elif len(feats_ft):
			X = X[feats_ft]

		rsp = 0
		if rspn != 'none':
			rsp = self.rsp_dict[rspn]
			X_rsp, y_rsp = rsp.fit_resample(X, y)
			X_rsp.index = X.index[rsp.sample_indices_]
			y_rsp.index = y.index[rsp.sample_indices_]
			X = X_rsp
			y = y_rsp

		return X, y, df, mc, ft, rsp

	def Predict(self, X_test, y_test, df, mc, ft, rsp, dn, mcn, ftn, rspn, is_verbose=1):
		t0 = time.time()

		y_pred  = mc.predict(X_test)
		y_score = mc.predict_proba(X_test)
		y_score = np.reshape(['%.2f' % sc for sc in np.ravel(y_score)], y_score.shape)

		if mcn == 'xgb':
			t_dict = {}
			for i, col in enumerate(y_test.columns): t_dict[str(i)] = col
			y_test = y_test.idxmax(axis=1)
			y_pred = np.array([t_dict[str(np.argmax(y))] for y in y_pred])

		acc = accuracy_score(y_test, y_pred)

		df_test = df.loc[y_test.index, self.params]
		df_test['type_p'] = y_pred
		df_test['score'] = list(map(str, y_score))
		
		t1 = time.time()
		if is_verbose:
			print('\n# Data : %s\n# Machine : %s\n# Feature selector : %s %d->%d\n# Resampler : %s\n# Accuracy : %f (%d/%d)\n# Elapsed Time : %fs\n'\
					% (dn, mcn, ftn, df.shape[1]-len(self.params), X_test.shape[1], rspn, acc, len(y_pred)-len(df_test[df_test['type'] != df_test['type_p']]), len(y_pred), t1-t0))
		return y_test, y_pred, y_score, df_test

	#def Train(self, dn, mcn, ftn, rspn, test_size=0.3, is_verbose=1, is_eps=0, is_checkset=0, tune_dict=0):
	def Train(self, dn, mcn, ftn, rspn, is_verbose=1, is_eps=0, test_size=0.3):
		if re.search('gen_', ftn): feats_ft = []
		elif ftn != 'none': feats_ft = np.loadtxt(self.path_save+ftn, dtype=str)
		else: feats_ft = []

		X, y, df, mc, ft, rsp = self.Preprocess(dn, mcn, ftn, rspn, feats_ft=feats_ft, is_eps=is_eps)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=self.random_state)

		"""
		if tune_dict:
			cv = StratifiedKFold(random_state=self.random_state)
			tn = RandomizedSearchCV(self.mc_dict[mcn], tune_dict, cv, random_state=self.random_state)
			res = tn.fit(X_train, y_train)
			return res
		if is_checkset:
			return X_train, X_test, y_train, y_test
		"""
		
		mc.fit(X_train, y_train)
		y_test, y_pred, y_score, df_test = self.Predict(X_test, y_test, df, mc, ft, rsp, dn, mcn, ftn, rspn, is_verbose=is_verbose)

		return y_test, y_pred, y_score, df_test, mc, ft, rsp

	def Validate(self, dn1, dn2, mcn, ftn, rspn, test_size=0.3, is_verbose=1, is_eps=0):
		if ftn != 'none': feats_ft = np.loadtxt(self.path_save+ftn, dtype=str)
		else: feats_ft = []

		y_test0, _, _, df1_test, mc, ft, rsp = self.Train(dn1, mcn, ftn, rspn, test_size=test_size, is_eps=is_eps, is_verbose=is_verbose)
		X_test, y_test, df2, _, _, _  = self.Preprocess(dn2, mcn, 'none', 'none', feats_ft=feats_ft, is_eps=is_eps)

		"""
		X, y, df2 = self.Preprocess(dn2, mcn, ftn, rspn, is_eps=is_eps)

		idx = y_test0.index
		if re.search('ddos', dn2):
			X_test = X
			y_test = y
		else:
			X_test = X.loc[idx, :]
			y_test = y.loc[idx] if mcn == 'xgb' else y[idx]
		"""

		y_test, y_pred, y_score, df2_test = self.Predict(X_test, y_test, df2, mc, ft, rsp, dn2, mcn, ftn, rspn, is_verbose=is_verbose)

		return y_test, y_pred, y_score, df1_test, df2_test, mc, ft, rsp

	def AnnotateHeatmap(self, im, data_txt=None, textcolors=("black", "white")):
		data = im.get_array()
		threshold = im.norm(data.max())/2.
		kw = dict(horizontalalignment="center", verticalalignment="center")

		texts = []
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
				text = im.axes.text(j, i, '%d' % data_txt[i, j], fontsize=32, **kw)
				texts.append(text)

		return texts

	def DrawConfMat(self, y_test, y_pred, ax):
		labels = list(self.type_dict)
		mat = confusion_matrix(y_test, y_pred, labels=labels)
		acc = [mat[i, :]/mat[i, :].sum() for i in range(3)]
		norm = plt.Normalize(0, 1)

		im = ax.imshow(acc, cmap='Blues', norm=norm)
		self.AnnotateHeatmap(im, data_txt=mat)
		
		return im

	def DrawDOS(self, dn, type, JU, N, U, ax, fermi_idx=0, point=0):
		path_dos = dn.split('/')[0]
		dtype    = dn.split('_')[0][0]
		ep       = re.sub('ep', '', re.search('ep\d[.]\d+', dn).group())
		bins     = int(re.sub('bins', '', re.search('bins\d+', dn).group()))

		JU = float(JU)
		N  = float(N)
		U  = float(U)
		fermi_idx = int(fermi_idx)
			
		pn = '%s/%s/params.txt' % (self.path_save, path_dos)
		with open(pn, 'r') as f:
			for idx, line in enumerate(f):
				if re.search('%d[.]\d+\s+%.1f\d+\s+%.1f\d+\s+%.1f\d+\s+' % (self.type_dict[type], JU, N, U), line):
					idx += fermi_idx - 1
					break

		with h5py.File(self.path_save+dn, 'r') as f: d = f['dos'][()]
		with h5py.File('%s/energy_bins%d.h5' % (self.path_save, bins), 'r') as f: e = f['energy'][()]

		x = d[idx][point*bins:(point+1)*bins]
		y = e

		peak = 0

		return x, y, peak
