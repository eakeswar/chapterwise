@echo off
cd /d "%~dp0..\.."
python backend\kaggle\publish_chapterwise_dataset.py
