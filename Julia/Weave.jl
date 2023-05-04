using Pkg: Pkg
Pkg.activate(".")
using Weave
weave("Introduction.jmd"; out_path="html/", fig_path=tempname())
weave("Lotka-Volterra.jmd"; out_path="html/", fig_path=tempname())
weave("SIR.jmd"; out_path="html/", fig_path=tempname())