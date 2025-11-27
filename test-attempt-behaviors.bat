@echo off
echo ====================================
echo Testing AuraBot & Feedback Behavior  
echo ====================================
echo.
echo This test verifies:
echo [1] AuraBot HIDDEN on attempts 0, 1, 2
echo [2] AuraBot SHOWS on attempts 3+
echo [3] Feedback auto-opens ONLY on 1st attempt
echo [4] Feedback stays closed on 2nd+ attempts
echo.

echo Step 1: Clear any existing submission data
powershell -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/admin/activities/1/clear-submissions' -Method POST; Write-Host '✅ Cleared submission history' -ForegroundColor Green } catch { Write-Host 'ℹ️ No submissions to clear or endpoint not available' -ForegroundColor Yellow }"

echo.
echo Step 2: Check activity status (should show 0 attempts)  
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/activities/1/status' -Method GET; Write-Host '📊 Activity Status:' -ForegroundColor Cyan; Write-Host \"   Total attempts: $($response.total_attempts)\" -ForegroundColor White; Write-Host \"   Completed: $($response.is_completed)\" -ForegroundColor White; if($response.total_attempts -eq 0) { Write-Host '✅ PASS: Starting with 0 attempts' -ForegroundColor Green } else { Write-Host '❌ FAIL: Should start with 0 attempts' -ForegroundColor Red } } catch { Write-Host '❌ Error checking activity status' -ForegroundColor Red }"

echo.
echo ====================================
echo 🎯 EXPECTED BEHAVIOR IN FRONTEND:
echo ====================================
echo.
echo For Activity 1 (http://localhost:3000/activity/1):
echo.
echo 💡 ATTEMPT 1:
echo    - AuraBot: HIDDEN ❌
echo    - Feedback: AUTO-OPENS after submit ✅
echo.  
echo 💡 ATTEMPT 2:
echo    - AuraBot: STILL HIDDEN ❌
echo    - Feedback: STAYS CLOSED after submit ❌ (click to open)
echo.
echo 💡 ATTEMPT 3:
echo    - AuraBot: NOW APPEARS! ✅
echo    - Feedback: STAYS CLOSED after submit ❌ (click to open)
echo.
echo 💡 ATTEMPT 4+:
echo    - AuraBot: STILL VISIBLE ✅  
echo    - Feedback: STAYS CLOSED after submit ❌ (click to open)
echo.
echo ====================================
echo 🧪 TEST INSTRUCTIONS:
echo ====================================
echo.
echo 1. Go to: http://localhost:3000/activity/1
echo 2. Submit any code 3 times
echo 3. Verify AuraBot appears only after 3rd attempt
echo 4. Verify Feedback auto-opens only on 1st attempt
echo.
pause
