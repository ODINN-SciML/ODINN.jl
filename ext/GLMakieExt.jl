module GLMakieExt

using GLMakie

# When GLMakie is loaded alongside ODINN, activate it as the backend so that
# 3D law plots (Axis3) become interactive and rotatable.
GLMakie.activate!()

end
