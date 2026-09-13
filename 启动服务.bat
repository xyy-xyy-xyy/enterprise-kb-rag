@echo off
chcp 65001 >nul
cd /d %~dp0

echo ========================================
echo   企业知识库 RAG - 启动脚本
echo ========================================
echo.

REM 检查虚拟环境
if not exist "venv\Scripts\python.exe" (
    echo [错误] 找不到虚拟环境 venv\Scripts\python.exe
    echo 请先创建虚拟环境：python -m venv venv
    pause
    exit /b 1
)

REM 检查索引是否存在
if not exist "data\faiss_index\index.faiss" (
    echo [提示] 未检测到 FAISS 索引，正在首次构建...
    echo.
    call venv\Scripts\python.exe -m app.ingestion
    echo.
    echo 索引构建完成。
    echo.
)

echo 正在启动服务...
echo 启动完成后，浏览器打开: http://localhost:8000
echo 按 Ctrl+C 停止服务
echo.

call venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --reload

pause
