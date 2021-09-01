@echo off
set condaRoot=D:\Program\Anaconda3
call %condaRoot%\Scripts\activate.bat
call conda activate pytorch-1-9
cd ./py
call python train.py

