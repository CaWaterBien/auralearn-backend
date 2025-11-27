@echo off
echo Testing Short AI Feedback
echo =========================
echo.

echo Testing with incomplete HTML code to see short feedback...
echo.

powershell -Command ^
  "$body = '{\"user_code\":\"^<html^>^<head^>^<title^>Test^</title^>^</head^>^<body^>^<h1^>Hello^</h1^>^</body^>^</html^>\",\"time_spent_minutes\":2}'; " ^
  "try { " ^
  "  $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/activities/2/submit' -Method POST -ContentType 'application/json' -Body $body; " ^
  "  Write-Host 'SUCCESS! Testing feedback length...' -ForegroundColor Green; " ^
  "  Write-Host 'Score:' $response.score -ForegroundColor Cyan; " ^
  "  Write-Host 'AI Powered:' $response.ai_powered -ForegroundColor Yellow; " ^
  "  Write-Host 'Feedback Length:' $response.feedback.Length 'characters' -ForegroundColor Magenta; " ^
  "  Write-Host '' -ForegroundColor White; " ^
  "  Write-Host 'FEEDBACK PREVIEW:' -ForegroundColor White; " ^
  "  Write-Host $response.feedback -ForegroundColor Gray; " ^
  "} catch { " ^
  "  Write-Host 'ERROR:' $_.Exception.Message -ForegroundColor Red; " ^
  "}"

echo.
echo =========================
echo Test Complete!
echo =========================
pause
