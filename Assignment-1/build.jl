using Pkg
using PackageCompiler
Pkg.instantiate()
create_app("AutoEncoder", "AECompiled")