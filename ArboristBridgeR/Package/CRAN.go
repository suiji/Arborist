//  Builds a CRAN-style temporary directory tree suitable for package
//  construction, then archives, then deletes the source tree.

package main

import (
       "os"
       "io"
       "io/ioutil"
       "fmt"
       "path/filepath"
)


func main() {
  srcFileInfo, err := os.Stat(".")
  topDest := "Rborist"

  _, err = os.Open(topDest)
  if !os.IsNotExist(err) {
    fmt.Print(topDest + " already exists")
    return
  }

  err = os.MkdirAll(topDest, srcFileInfo.Mode())
  if err != nil {
    fmt.Printf("Cannot generate top-level directory " + topDest)
    return
  }

  err = CopyFile(".." + string(filepath.Separator) + "LICENSE", topDest)
  if err != nil {
     fmt.Printf("Copy error")
  }

  feSource := ".." + string(filepath.Separator) + "FrontEnd" + string(filepath.Separator)
  err = CopyFile(feSource + "DESCRIPTION", topDest)
  err = CopyFile(feSource + "NAMESPACE", topDest)

  manDest := topDest + string(filepath.Separator) + "man"
  err = os.MkdirAll(manDest, srcFileInfo.Mode())
  err = CopyFile(feSource + "*.Rd", manDest)

  RDest := topDest + string(filepath.Separator) + "R"
  os.MkdirAll(RDest, srcFileInfo.Mode())
  err = CopyFile(feSource + "*R", RDest)

  coreSource := ".." + string(filepath.Separator) + "ArboristCore" + string(filepath.Separator)
  sharedSource := ".." + string(filepath.Separator) + "Shared" + string(filepath.Separator)
  sourceDest := topDest + string(filepath.Separator) + "src" + string(filepath.Separator)
  os.MkdirAll(sourceDest, srcFileInfo.Mode())
  err = CopyFile(coreSource + "*", sourceDest)
  err = CopyFile(sharedSource + "*", sourceDest)

// Deep copy:
  testSource := ".." + string(filepath.Separator) + "tests"
  CopyTree(testSource, topDest)

  //  R CMD build Rborist
    //os.RemoveAll(topDest)
}


// Adapted from Github project termie/go-shutil:

// Copy data from src to dst
//
func CopyFile(src string, dst string) (error) {
  _, err := os.Stat(dst)
  if err != nil && !os.IsNotExist(err) {
    return err
  }


  // Do the actual copy
  fsrc, err := os.Open(src)
  if err != nil {
    return err
  }
  defer fsrc.Close()

  fdst, err := os.Create(dst)
  if err != nil {
    return err
  }
  defer fdst.Close()

  _, err = io.Copy(fdst, fsrc)
  if err != nil {
    return err
  }


  return nil
}


// Recursively copy a directory tree.
//
// The destination directory must not already exist.
//
// Since CopyTree() is called recursively, the callable will be
// called once for each directory that is copied. It returns a
// list of names relative to the `src` directory that should
// not be copied.
//
func CopyTree(src, dst string) error {

  srcFileInfo, err := os.Stat(src)
  if err != nil {
    return err
  }

  entries, err := ioutil.ReadDir(src)
  if err != nil {
    return err
  }

  err = os.MkdirAll(dst, srcFileInfo.Mode())
  if err != nil {
    return err
  }

  for _, entry := range entries {
    srcPath := filepath.Join(src, entry.Name())
    dstPath := filepath.Join(dst, entry.Name())

    entryFileInfo, err := os.Lstat(srcPath)
    if err != nil {
      return err
    }

    if entryFileInfo.IsDir() {
      err = CopyTree(srcPath, dstPath)
    } else {
      err = CopyFile(srcPath, dstPath)
    }
    if (err != nil) {
       return err
    }
  }
  return nil
}