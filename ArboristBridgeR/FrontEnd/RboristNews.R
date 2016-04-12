RboristNews <- function() {
    newsfile <- file.path(system.file(package="Rborist"), "NEWS")
    file.show(newsfile)
}
