<?php

namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Validator;
use Illuminate\Validation\ValidationException;

class UserAuthController extends Controller
{
    /**
     * Register a new user
     */
    public function register(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'fullName' => ['required', 'string', 'max:255'],
            'email' => ['required', 'string', 'email', 'max:255', 'unique:users'],
            'password' => ['required', 'string', 'min:8'],
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $user = new User();
            $user->name = $request->fullName;
            $user->full_name = $request->fullName;
            $user->email = $request->email;
            $user->password = Hash::make($request->password);
            $user->join_date = now()->toDateString();
            $user->is_active = 1;
            $user->save();
            
            // Set JSON fields after saving
            $user->progress = $user->getDefaultProgress();
            $user->preferences = $user->getDefaultPreferences();
            $user->save();

            // Log the user in
            Auth::login($user);

            return response()->json([
                'message' => 'Registration successful',
                'user' => [
                    'id' => $user->id,
                    'fullName' => $user->full_name,
                    'email' => $user->email,
                    'avatar' => $user->avatar,
                    'joinDate' => $user->join_date,
                    'progress' => $user->progress,
                    'preferences' => $user->preferences,
                ]
            ], 201);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Registration failed',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Login user
     */
    public function login(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'email' => ['required', 'string', 'email'],
            'password' => ['required', 'string'],
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        $user = User::where('email', $request->email)->first();

        if (!$user || !Hash::check($request->password, $user->password)) {
            return response()->json([
                'message' => 'Invalid credentials'
            ], 401);
        }

        if ($user->is_active !== 1) {
            return response()->json([
                'message' => 'Account is deactivated'
            ], 403);
        }

        // Log the user in
        Auth::login($user, $request->remember ?? false);

        return response()->json([
            'message' => 'Login successful',
            'user' => [
                'id' => $user->id,
                'fullName' => $user->full_name,
                'email' => $user->email,
                'avatar' => $user->avatar,
                'joinDate' => $user->join_date,
                'progress' => $user->progress,
                'preferences' => $user->preferences,
            ]
        ]);
    }

    /**
     * Get authenticated user
     */
    public function me(Request $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        return response()->json([
            'user' => [
                'id' => $user->id,
                'fullName' => $user->full_name,
                'email' => $user->email,
                'avatar' => $user->avatar,
                'joinDate' => $user->join_date,
                'progress' => $user->progress,
                'preferences' => $user->preferences,
            ]
        ]);
    }

    /**
     * Logout user
     */
    public function logout(Request $request)
    {
        Auth::logout();
        $request->session()->invalidate();
        $request->session()->regenerateToken();

        return response()->json([
            'message' => 'Logout successful'
        ]);
    }

    /**
     * Update user profile
     */
    public function updateProfile(Request $request)
    {
        $user = Auth::user();
        
        if (!$user) {
            return response()->json([
                'message' => 'Unauthenticated'
            ], 401);
        }

        $validator = Validator::make($request->all(), [
            'fullName' => ['sometimes', 'string', 'max:255'],
            'avatar' => ['sometimes', 'string', 'max:500'],
            'preferences' => ['sometimes', 'array'],
            'progress' => ['sometimes', 'array'],
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $updateData = [];
            
            if ($request->has('fullName')) {
                $updateData['name'] = $request->fullName;
                $updateData['full_name'] = $request->fullName;
            }
            
            if ($request->has('avatar')) {
                $updateData['avatar'] = $request->avatar;
            }
            
            if ($request->has('preferences')) {
                $updateData['preferences'] = $request->preferences;
            }
            
            if ($request->has('progress')) {
                $updateData['progress'] = $request->progress;
            }

            $user->update($updateData);

            return response()->json([
                'message' => 'Profile updated successfully',
                'user' => [
                    'id' => $user->id,
                    'fullName' => $user->full_name,
                    'email' => $user->email,
                    'avatar' => $user->avatar,
                    'joinDate' => $user->join_date,
                    'progress' => $user->progress,
                    'preferences' => $user->preferences,
                ]
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Profile update failed',
                'error' => $e->getMessage()
            ], 500);
        }
    }
}