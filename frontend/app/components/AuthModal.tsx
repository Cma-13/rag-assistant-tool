"use client";

import { useState } from "react";
import { GoogleOAuthProvider, GoogleLogin } from "@react-oauth/google";

// The Client ID is public (it is not a secret), so it is safe to keep here.
const GOOGLE_CLIENT_ID =
  process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ??
  "972255011251-1cok0vt59jiolcut4a8v74mump93vmb6.apps.googleusercontent.com";

interface AuthModalProps {
  onClose: () => void;
  onAuthSuccess: (token: string, email: string) => void;
  initialMode?: "login" | "signup";
}

export default function AuthModal({ onClose, onAuthSuccess, initialMode = "login" }: AuthModalProps) {
  const [mode, setMode] = useState<"login" | "signup">(initialMode);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!email.trim() || !password.trim()) {
      setError("Please enter both email and password.");
      return;
    }
    setError("");
    setLoading(true);

    try {
      const endpoint = mode === "login" ? "/login" : "/signup";
      const res = await fetch(`http://127.0.0.1:8000${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.detail ?? "Something went wrong. Please try again.");
        setLoading(false);
        return;
      }

      onAuthSuccess(data.access_token, email);
    } catch {
      setError("Couldn't reach the server. Please try again.");
      setLoading(false);
    }
  };

  const handleGoogle = async (credential?: string) => {
    if (!credential) {
      setError("Google sign-in failed. Please try again.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/auth/google", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ credential }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail ?? "Google sign-in failed. Please try again.");
        setLoading(false);
        return;
      }
      onAuthSuccess(data.access_token, data.email);
    } catch {
      setError("Couldn't reach the server. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
      <div className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-xl">
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-[#1C1C1E]">
            {mode === "login" ? "Log in" : "Create an account"}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-[#9A968C] hover:text-[#1C1C1E]"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <form
          className="space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            handleSubmit();
          }}
        >
          <input
            type="email"
            name="email"
            autoComplete="username"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-xl border border-[#E6E0D2] px-3 py-2 text-[14.5px] outline-none focus:border-[#5E7C5C]"
          />
          <input
            type="password"
            name="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-xl border border-[#E6E0D2] px-3 py-2 text-[14.5px] outline-none focus:border-[#5E7C5C]"
          />

          {error && <p className="text-[13px] text-red-600">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-xl bg-[#5E7C5C] py-2 text-[14.5px] font-medium text-white transition hover:bg-[#4E684C] disabled:opacity-50"
          >
            {loading
              ? "Please wait…"
              : mode === "login"
                ? "Log in"
                : "Sign up"}
          </button>
        </form>

        <div className="my-4 flex items-center gap-3 text-[12px] text-[#9A968C]">
          <span className="h-px flex-1 bg-[#E0DBD3]" />
          or
          <span className="h-px flex-1 bg-[#E0DBD3]" />
        </div>

        <div className="flex justify-center">
          <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
            <GoogleLogin
              onSuccess={(res) => handleGoogle(res.credential)}
              onError={() => setError("Google sign-in failed. Please try again.")}
              text={mode === "login" ? "continue_with" : "signup_with"}
              width="300"
            />
          </GoogleOAuthProvider>
        </div>

        <p className="mt-4 text-center text-[13px] text-[#6B6B6B]">
          {mode === "login" ? (
            <>
              Don&apos;t have an account?{" "}
              <button
                type="button"
                onClick={() => {
                  setMode("signup");
                  setError("");
                }}
                className="font-medium text-[#5E7C5C] hover:underline"
              >
                Sign up
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button
                type="button"
                onClick={() => {
                  setMode("login");
                  setError("");
                }}
                className="font-medium text-[#5E7C5C] hover:underline"
              >
                Log in
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}