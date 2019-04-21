.onAttach <- function(libname, pkgname) {
    RbVer <- read.dcf(file=system.file("DESCRIPTION", package=pkgname),
                      fields="Version")
    packageStartupMessage(paste(pkgname, RbVer))
    packageStartupMessage("Type RboristNews() to see new features/changes/bug fixes.")
}
