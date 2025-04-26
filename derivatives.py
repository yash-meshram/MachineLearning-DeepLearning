import sympy

J, w = sympy.symbols("J, w")
x = 2

J = w**2
dJ_dw = sympy.diff(J, w)
print(f"J = {J}\ndJ_dw = {dJ_dw}\ndJ_dw = {dJ_dw.subs([(w, x)])} for w = {x}\n")

J = w**3
dJ_dw = sympy.diff(J, w)
print(f"J = {J}\ndJ_dw = {dJ_dw}\ndJ_dw = {dJ_dw.subs([(w, x)])} for w = {x}\n")

J = 1/w
dJ_dw = sympy.diff(J, w)
print(f"J = {J}\ndJ_dw = {dJ_dw}\ndJ_dw = {dJ_dw.subs([(w, x)])} for w = {x}\n")

J = w
dJ_dw = sympy.diff(J, w)
print(f"J = {J}\ndJ_dw = {dJ_dw}\ndJ_dw = {dJ_dw.subs([(w, x)])} for w = {x}\n")
