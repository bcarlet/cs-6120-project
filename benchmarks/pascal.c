#define N 10

void pascal(int a[static N][N]) {
    for (int i = 0; i < N; i++) {
        a[i][0] = 1;

        for (int j = 1; j <= i; j++) {
            a[i][j] = a[i - 1][j - 1] + a[i - 1][j];
        }
    }
}
