# Setup
You must have an install of Weka on your machine.
https://ml.cms.waikato.ac.nz/weka/downloading.html

In the Weka GUI package manager install the Normalize package. Along with this, if you intend to use Artificial Immune System (AIS) models, the following files must also be installed.
https://github.com/fracpete/wekaclassalgos

From the Weka install, the Weka.jar must be linked as a library in your project. The same must be done with the wekaclassalgos and Normalize JAR files. Weka should have installed the packages to '%userprofile%\wekafiles\packages'.

You will have to add '--add-opens java.base/java.lang=ALL-UNNAMED' as a VM option for Java versions past 8.
