import sys

import islpy as isl
import sympy as sp

def extract_matrices(domain):
    mat = domain.inequalities_matrix(
        isl.dim_type.set,
        isl.dim_type.param,
        isl.dim_type.cst,
        isl.dim_type.div,
    )

    dims = domain.get_var_names(isl.dim_type.set)
    params = domain.get_var_names(isl.dim_type.param)

    dimc = len(dims)
    varc = len(dims) + len(params)

    dims_mat = sp.Matrix([
        [mat.get_element_val(i, j).to_python()
            for j in range(dimc)]
                for i in range(mat.rows())
    ])

    const_mat = sp.Matrix([
        mat.get_element_val(i, varc).to_python()
            for i in range(mat.rows())
    ])

    if dimc != varc:
        params_mat = sp.Matrix([
            [mat.get_element_val(i, j).to_python()
                for j in range(dimc, varc)]
                    for i in range(mat.rows())
        ])

        const_mat += params_mat * sp.Matrix(params)

    return dims_mat, sp.Matrix(dims), const_mat

def main():
    source = sys.stdin.read()
    domain = isl.BasicSet(source.replace('%', ''))

    const_mat, iter_vec, const_vec = extract_matrices(domain)

    print(
        sp.latex(const_mat),
        sp.latex(iter_vec),
        '+',
        sp.latex(const_vec),
        r'\geq \mathbf{0}'
    )

if __name__ == '__main__':
    main()
