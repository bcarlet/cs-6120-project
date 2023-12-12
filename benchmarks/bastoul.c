/**
 * Examples adapted from https://icps.u-strasbg.fr/people/bastoul/public_html/research/papers/Bastoul_thesis.pdf
 */

#define M 4
#define N 4

void fig_2_3(int a[restrict static N], int b[restrict static N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i <= N + 2 - j)
                b[j] += a[i];
        }
    }
}

void fig_2_4(int x[static 10]) {
    for (int i = 1; i <= 3; i++) {
        x[i] += 1;      // S1

        for (int j = 1; j <= i * i; j++) {
            x[j] += 2;  // S2

            for (int k = 0; k <= j; k++) {
                if (j >= 2)
                    x[k] += 3;  // S3

                x[k] += 4;      // S4
            }
        }

        for (int p = 0; p <= 6; p++) {
            x[p] += 5;  // S5
            x[p] += 6;  // S6
        }
    }
}

void fig_3_8(int a[restrict static N], int b[restrict static M]) {
    for (int i = 0; i < N; i++) {
        a[i] = i;   // S1

        for (int j = 0; j < M; j++) {
            b[j] = (b[j] + a[i]) / 2;   // S2
        }
    }
}
