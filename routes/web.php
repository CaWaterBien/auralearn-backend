<?php

use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return view('welcome');
});

// Quick connectivity check route: GET /supabase/ping
Route::get('/supabase/ping', function () {
    try {
        // Simple connectivity test using Laravel's DB facade
        \DB::connection('pgsql')->select('SELECT 1 as test');
        return response()->json(['ok' => true]);
    } catch (\Throwable $e) {
        return response()->json(['ok' => false, 'error' => $e->getMessage()], 500);
    }
});
