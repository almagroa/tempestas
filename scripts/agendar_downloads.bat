@echo off
REM Cria uma tarefa diária no Agendador do Windows que atualiza os dados do Tempestas às 07h.
REM Rode uma vez, no prompt do Anaconda, dentro da pasta do repositório.
set PY=%CONDA_PREFIX%\python.exe
if "%CONDA_PREFIX%"=="" set PY=python
schtasks /Create /F /SC DAILY /ST 07:00 /TN "Tempestas - downloads" ^
  /TR "\"%PY%\" -m tempestas baixar" /RL LIMITED
echo Tarefa "Tempestas - downloads" criada (diaria, 07:00).
