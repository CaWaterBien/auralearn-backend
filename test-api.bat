@echo off
echo Testing AuraBot API endpoints...
echo ================================

timeout /t 3 /nobreak > nul

echo Testing health endpoint...
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/health' -Method GET; Write-Host 'SUCCESS: Health endpoint working!'; $response } catch { Write-Host ('Health endpoint ERROR: ' + $_.Exception.Message) }"

echo.
echo Testing session status...
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/session-status?session_id=test123' -Method GET; Write-Host 'SUCCESS: Session status working!'; $response } catch { Write-Host ('Session status ERROR: ' + $_.Exception.Message) }"

echo.
echo Testing ask question (this will fail without OpenAI key but should reach the endpoint)...
powershell -Command "try { $body = @{ session_id='test123'; question='What is HTML?' } | ConvertTo-Json; $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/ask' -Method POST -Body $body -ContentType 'application/json'; Write-Host 'SUCCESS: Ask endpoint working!'; $response } catch { Write-Host ('Ask endpoint result: ' + $_.Exception.Message) }"

echo.
echo ================================
echo API Testing Complete!
pause
