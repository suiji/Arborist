AC_INIT([Rborist], 0.1-0)

AC_LANG(C++)
AC_REQUIRE_CPP


AC_MSG_NOTICE([checking for critical tools used by configure])


# Much of what follows has been pilferred from RQuantLib package
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
    AC_MSG_ERROR([Could not determine R_HOME.])   
fi

#R_INCL=`"${R_HOME}/bin/R" CMD config --cppflags`
#R_LIB=`"${R_HOME}/bin/R" CMD config --ldflags`
#R_CPIC=`"${R_HOME}/bin/R" CMD config CPICFLAGS`

## look for Rscript, but use the one found via R_HOME to allow for multiple installations
AC_DEFUN(AC_PROG_RSCRIPT, [AC_CHECK_PROG(RSCRIPT,Rscript,yes)])
AC_PROG_RSCRIPT

if test x"${RSCRIPT}" == x"no" ; then
#     rcpp_ldflags=`${R_HOME}/bin/Rscript -e 'Rcpp:::LdFlags()'`
#     AC_SUBST([RCPP_LDFLAGS],["${LDFLAGS} $rcpp_ldflags"])
#     rcpp_incl=`${R_HOME}/bin/Rscript -e 'Rcpp:::CxxFlags()'`
#     AC_SUBST([RCPP_INCL],["${LDFLAGS} $rcpp_incl"])
#else
    echo "
  Your installation does not appear to have Rscript installed.

  Please make sure that you have a working and complete R installation.
"
    exit 1
fi


AC_MSG_NOTICE([determining host platform])
arch=`uname -m`
osrelease=`uname -r`
osversion=`uname -v`

AC_MSG_NOTICE([Building Makefile])
#AC_SUBST(R_INCL)
#AC_SUBST(R_LIB)
#AC_SUBST(R_CPIC)
#AC_SUBST(RCPP_LDFLAGS)
#AC_SUBST(RCPP_INCL)
AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT
