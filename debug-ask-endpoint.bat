@echo off
echo ================================
echo   DEBUGGING ASK ENDPOINT
echo ================================

echo 1. Testing ask endpoint with PowerShell...
powershell -Command ^
"try { ^
  $body = @{ ^
    session_id = 'test_debug_session'; ^
    question = 'What is HTML?'; ^
    html_context = $null; ^
    instructions_context = $null ^
  } | ConvertTo-Json; ^
  Write-Host 'Sending request body:' $body; ^
  $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/ask' -Method POST -Body $body -ContentType 'application/json'; ^
  Write-Host 'SUCCESS! Response received:'; ^
  Write-Host 'Success:' $response.success; ^
  Write-Host 'Response:' $response.response; ^
} catch { ^
  Write-Host 'ERROR Details:' $_.Exception.Message; ^
  Write-Host 'Status Code:' $_.Exception.Response.StatusCode; ^
}"

echo.
echo 2. Checking recent logs...
powershell -Command "Get-Content storage\logs\laravel.log -Tail 10 | Where-Object {$_ -match 'ERROR\|Exception\|Fatal'}"

pause
