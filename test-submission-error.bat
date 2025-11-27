@echo off
echo Testing Activity Submission to Find 500 Error
echo =============================================
echo.

echo Step 1: Testing submission endpoint
powershell -Command ^
  "$body = '{\"user_code\":\"^<html^>^<head^>^<title^>Test^</title^>^</head^>^<body^>^<h1^>Hello^</h1^>^</body^>^</html^>\",\"time_spent_minutes\":5}'; " ^
  "try { " ^
  "  $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/activities/5/submit' -Method POST -ContentType 'application/json' -Body $body; " ^
  "  Write-Host 'SUCCESS:' $response.score -ForegroundColor Green; " ^
  "  Write-Host 'AI Powered:' $response.ai_powered -ForegroundColor Cyan; " ^
  "} catch { " ^
  "  Write-Host 'ERROR:' $_.Exception.Message -ForegroundColor Red; " ^
  "  if ($_.Exception.Response) { " ^
  "    Write-Host 'Status Code:' $_.Exception.Response.StatusCode -ForegroundColor Yellow; " ^
  "  } " ^
  "}"

echo.
echo Step 2: Checking Laravel logs for errors
if exist "storage\logs\laravel.log" (
    echo Found Laravel log - showing last 10 lines:
    powershell -Command "Get-Content storage\logs\laravel.log | Select-Object -Last 10"
) else (
    echo No Laravel log found
)

echo.
echo Test completed!
pause
