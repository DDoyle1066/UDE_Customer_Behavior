using Pkg: Pkg
Pkg.activate(".")
using Weave
weave("Introduction.jmd"; out_path="html/", fig_path=tempname())
weave("Latka-Volterra.jmd"; out_path="html/", fig_path=tempname())