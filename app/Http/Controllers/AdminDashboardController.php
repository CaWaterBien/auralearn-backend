<?php

namespace App\Http\Controllers;

use App\Models\User;
use App\Models\Admin;
use Illuminate\Http\Request;

class AdminDashboardController extends Controller
{
    public function overview(Request $request)
    {
        $usersCount = User::count();
        $adminsCount = Admin::count();
        $jobsCount = \DB::table('jobs')->count();

        $supabaseOk = true;
        $supabaseError = null;
        try {
            // Simple connectivity test - just check if the database connection works
            // Using a simple query that doesn't depend on specific tables existing
            \DB::connection('pgsql')->select('SELECT 1 as test');
        } catch (\Throwable $e) {
            $supabaseOk = false;
            $supabaseError = $e->getMessage();
        }

        return response()->json([
            'stats' => [
                'users' => $usersCount,
                'admins' => $adminsCount,
                'queuedJobs' => $jobsCount,
                'supabase' => [
                    'ok' => $supabaseOk,
                    'error' => $supabaseError,
                ],
            ],
        ]);
    }

    public function users(Request $request)
    {
        $limit = (int) min((int) $request->query('limit', 20), 100);
        $users = User::orderByDesc('id')->limit($limit)->get(['id','name','email','created_at']);
        return response()->json([
            'users' => $users,
        ]);
    }

    public function logs(Request $request)
    {
        $limit = (int) min((int) $request->query('limit', 200), 2000);
        $path = storage_path('logs/laravel.log');
        if (!file_exists($path)) {
            return response()->json(['lines' => []]);
        }
        $lines = [];
        $fp = fopen($path, 'r');
        if ($fp) {
            // Efficient tail-like read
            $buffer = '';
            while (!feof($fp)) {
                $buffer .= fread($fp, 8192);
            }
            fclose($fp);
            $all = preg_split("/\r?\n/", $buffer);
            $all = array_values(array_filter($all, fn($l) => $l !== ''));
            $lines = array_slice($all, -$limit);
        }
        return response()->json(['lines' => $lines]);
    }

    public function settings(Request $request)
    {
        $env = [
            'APP_NAME' => env('APP_NAME'),
            'APP_ENV' => env('APP_ENV'),
            'APP_URL' => env('APP_URL'),
            'DB_CONNECTION' => env('DB_CONNECTION'),
            'CACHE_STORE' => env('CACHE_STORE'),
            'QUEUE_CONNECTION' => env('QUEUE_CONNECTION'),
            'FILESYSTEM_DISK' => env('FILESYSTEM_DISK'),
            'MAIL_MAILER' => env('MAIL_MAILER'),
            'SESSION_DRIVER' => env('SESSION_DRIVER'),
        ];
        return response()->json(['settings' => $env]);
    }
} 