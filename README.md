# TARI29
Group assignment




## Setting Up Your LaTeX Environment in VS Code

To write and compile LaTeX documents within Visual Studio Code, you can install the necessary extensions via the command line.

### 1. Install Visual Studio Code
First, ensure that you have [Visual Studio Code](https://code.visualstudio.com/) installed on your system.

### 2. Install LaTeX Workshop Extension

To install the LaTeX Workshop extension, open your terminal or command prompt and run the following command:

```bash
code --install-extension James-Yu.latex-workshop
```
### 3. Install MiKTeX

Go to [MiKTeX](https://miktex.org/download/) and download installer.

### 4. Install perl.exe.

Go to [perl](https://strawberryperl.com/) and download installer.

### 5. Restart VSC and CMD

Restart Visual Studio Code and your terminals for the new paths to update. If errors such as "bla bla is not recognized" go check your paths.

!!! IF PROBLEMS !!! follow this [tutorial](https://www.youtube.com/watch?v=4lyHIQl4VM8).

Save and compile with CTRL + S should work.

Compiling will generate some garbage files. You should be able to add -c command to latexmk compile command like
```bash
latexmk -c
```
which will remove the garbage files.