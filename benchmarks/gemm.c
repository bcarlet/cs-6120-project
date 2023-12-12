#define M 200
#define N 300
#define P 400

void matmul(float a[M][N], float b[N][P], float c[M][P]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
