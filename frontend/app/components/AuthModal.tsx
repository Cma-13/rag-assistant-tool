"use client";

import { useState } from "react";
import { GoogleOAuthProvider, GoogleLogin } from "@react-oauth/google";
import { API_URL } from "../config";

// The Client ID is public (it is not a secret), so it is safe to keep here.
const GOOGLE_CLIENT_ID =
  process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ??
  "972255011251-1cok0vt59jiolcut4a8v74mump93vmb6.apps.googleusercontent.com";

// The rules an email must follow to sign up (the server checks the same rules).
// The text is trimmed and lower-cased first, then it must pass all of these.
const EMAIL_ERROR = "Please enter a valid email address.";

function emailProblem(raw: string): string | null {
  const email = raw.trim().toLowerCase();
  if (!email) return EMAIL_ERROR;
  if (/\s/.test(email)) return EMAIL_ERROR;
  if (email.length > 254) return EMAIL_ERROR;
  const at = email.split("@");
  if (at.length < 2) return EMAIL_ERROR;
  if (at.length > 2) return EMAIL_ERROR;
  const [local, domain] = at;
  if (!local) return EMAIL_ERROR;
  if (local.length > 64) return EMAIL_ERROR;
  if (!/^[a-z0-9._%+-]+$/.test(local)) {
    return EMAIL_ERROR;
  }
  if (local.startsWith(".") || local.endsWith(".")) return EMAIL_ERROR;
  if (local.includes("..")) return EMAIL_ERROR;
  if (!domain || domain.length < 3) return EMAIL_ERROR;
  if (domain.length > 255) return EMAIL_ERROR;
  if (!/^[a-z0-9.-]+$/.test(domain)) return EMAIL_ERROR;
  const labels = domain.split(".");
  if (labels.length < 2) return EMAIL_ERROR;
  if (labels.some((l) => l === "" || l.length > 63)) return EMAIL_ERROR;
  if (labels.some((l) => l.startsWith("-") || l.endsWith("-"))) {
    return EMAIL_ERROR;
  }
  if (!/^[a-z]{2,}$/.test(labels[labels.length - 1])) return EMAIL_ERROR;
  return null;
}

const MIN_PASSWORD_LENGTH = 8;

function EyeIcon({ off }: { off: boolean }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {off ? (
        <>
          <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
          <line x1="1" y1="1" x2="23" y2="23" />
        </>
      ) : (
        <>
          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
          <circle cx="12" cy="12" r="3" />
        </>
      )}
    </svg>
  );
}

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
  const [showPassword, setShowPassword] = useState(false);
  // A box only starts showing its red message after the user has left it (or pressed the button),
  // so nobody is told "wrong" while they are still typing. After that it updates as they fix it.
  const [emailTouched, setEmailTouched] = useState(false);
  const [passwordTouched, setPasswordTouched] = useState(false);

  // The problems are shown in red (in both Log in and Sign up) once the user has finished a box.
  // Only signing up is blocked by them: an account made before these rules existed must still
  // be able to log in, so there the server decides.
  const signingUp = mode === "signup";
  const emailHint = emailTouched && email.trim() ? emailProblem(email) : null;
  const passwordTooShort = passwordTouched && password.length > 0 && password.length < MIN_PASSWORD_LENGTH;

  const handleSubmit = async () => {
    setEmailTouched(true);
    setPasswordTouched(true);
    if (!email.trim() || !password.trim()) {
      setError("Please enter both email and password.");
      return;
    }
    if (signingUp) {
      // the red messages under the boxes already say what is wrong, so just stop here
      if (emailProblem(email) || password.length < MIN_PASSWORD_LENGTH) {
        setError("");
        return;
      }
    }
    setError("");
    setLoading(true);

    try {
      const endpoint = mode === "login" ? "/login" : "/signup";
      const res = await fetch(`${API_URL}${endpoint}`, {
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
      const res = await fetch(`${API_URL}/auth/google`, {
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
          noValidate
          onSubmit={(e) => {
            e.preventDefault();
            handleSubmit();
          }}
        >
          <div>
            <input
              type="email"
              name="email"
              autoComplete="username"
              placeholder="Email"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                setError("");
              }}
              onBlur={() => setEmailTouched(true)}
              className={`w-full rounded-xl border px-3 py-2 text-[14.5px] outline-none focus:border-[#5E7C5C] ${
                emailHint ? "border-red-400" : "border-[#E6E0D2]"
              }`}
            />
            {emailHint && <p className="mt-1 text-[12.5px] text-red-600">{emailHint}</p>}
          </div>
          <div>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                placeholder="Password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError("");
                }}
                onBlur={() => setPasswordTouched(true)}
                className={`w-full rounded-xl border py-2 pl-3 pr-10 text-[14.5px] outline-none focus:border-[#5E7C5C] ${
                  passwordTooShort ? "border-red-400" : "border-[#E6E0D2]"
                }`}
              />
              <button
                type="button"
                onClick={() => setShowPassword((s) => !s)}
                onMouseDown={(e) => e.preventDefault()} // keep the cursor in the password box
                aria-label={showPassword ? "Hide password" : "Show password"}
                title={showPassword ? "Hide password" : "Show password"}
                className="absolute right-2 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-md text-[#9A968C] hover:text-[#5E7C5C]"
              >
                <EyeIcon off={showPassword} />
              </button>
            </div>
            {/* only shown when the password is too short */}
            {passwordTooShort && (
              <p className="mt-1 text-[12.5px] text-red-600">At least {MIN_PASSWORD_LENGTH} characters</p>
            )}
          </div>

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
          <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID} locale="en">
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