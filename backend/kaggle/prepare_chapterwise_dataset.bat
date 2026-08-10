@echo off
REM Build chapterwise-models-bundle for Kaggle dataset chapterwise-models
setlocal
set DEST=%~dp0chapterwise-models-bundle

if exist "%DEST%" rmdir /S /Q "%DEST%"
mkdir "%DEST%" 2>nul

copy /Y "%~dp0..\security.py" "%DEST%\"
copy /Y "%~dp0..\pdf_extract.py" "%DEST%\"
copy /Y "%~dp0chapterwise_gpu_server.py" "%DEST%\"
copy /Y "%~dp0llm_server.py" "%DEST%\"

echo.
echo Bundle ready: %DEST%
echo Publish: backend\kaggle\publish_chapterwise_dataset.bat
