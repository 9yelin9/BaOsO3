#define USE_MATH_DEFINES

#define Nb 12

#include <omp.h>
#include <math.h>
#include <hdf5.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <lapack.h>
#include <sys/stat.h>

void GenKDOS(int Nk, int bins, double *e, double *eps, double *ws, double *data, double *dos) {
	int i, j, k;

	for(i=0; i<Nk; i++) {
		for(j=0; j<bins; j++) {
			dos[bins*i + j] = 0;
			for(k=0; k<Nb; k++) {
				dos[bins*i + j] += (eps[j] / (pow(e[j] - data[Nb*(2*i)+k], 2) + pow(eps[j], 2))) * data[Nb*(2*i+1)+k] * ws[j];
			}
			dos[bins*i + j] /= M_PI;
		}
	}
}
