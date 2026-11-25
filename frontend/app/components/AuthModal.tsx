"use client";

import { useEffect, useState } from "react";
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

function LogoMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M13 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h4" />
      <path d="M13 3l5 5v3" />
      <path d="M13 3v4a1 1 0 0 0 1 1h4" />
      <line x1="8.5" y1="9.5" x2="11" y2="9.5" />
      <line x1="8.5" y1="13" x2="12" y2="13" />
      <circle cx="16" cy="16.5" r="3" />
      <line x1="18.2" y1="18.7" x2="20.5" y2="21" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="flex-shrink-0">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
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

  const inputClass = (bad: boolean) =>
    `w-full rounded-xl border bg-white px-3.5 py-2.5 text-[14.5px] text-[#1C1C1E] outline-none transition placeholder:text-[#B5B0A5] focus:shadow-[0_0_0_4px_rgba(94,124,92,0.12)] ${
      bad ? "border-red-400 focus:border-red-400" : "border-[#E6E0D2] hover:border-[#D6CFBF] focus:border-[#5E7C5C]"
    }`;

  // Esc closes the window, like the other pop-ups (not while signing in)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !loading) onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [loading, onClose]);

  const switchMode = (next: "login" | "signup") => {
    setMode(next);
    setError("");
  };

  return (
    <div
      className="auth-backdrop fixed inset-0 z-50 flex items-center justify-center bg-[#1C1C1E]/40 px-4 py-6 backdrop-blur-[3px]"
      // a click on the dimmed background (not inside the window) closes it
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !loading) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-title"
        className="auth-card relative flex max-h-full w-full max-w-[400px] overflow-hidden rounded-3xl bg-white shadow-2xl md:max-w-[840px]"
      >
        {/* Brand panel (wide screens only): what DocQuery does */}
        <aside className="relative hidden w-[42%] flex-col justify-between overflow-hidden bg-[#5E7C5C] p-8 text-[#FEFAF3] md:flex">
          <div className="pointer-events-none absolute -right-24 -top-24 h-64 w-64 rounded-full bg-white/[0.06]" />
          <div className="pointer-events-none absolute -bottom-32 -left-20 h-72 w-72 rounded-full bg-black/[0.07]" />

          <div className="relative flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#FEFAF3] text-[#5E7C5C]">
              <LogoMark />
            </span>
            <span className="text-[15px] font-semibold tracking-tight">DocQuery</span>
          </div>

          <div className="relative">
            <p className="text-[26px] font-semibold leading-[1.2] tracking-tight">
              Ask your PDFs.
              <br />
              Get answers with the page they came from.
            </p>
          </div>

          <p className="relative text-[12.5px] leading-relaxed text-[#FEFAF3]/75">
            Your documents stay private to your account.
          </p>
        </aside>

        {/* Form side */}
        <div className="flex-1 overflow-y-auto px-6 py-7 sm:px-10 sm:py-9">
          <button
            type="button"
            onClick={onClose}
            className="absolute right-4 top-4 flex h-9 w-9 items-center justify-center rounded-full text-[#9A968C] transition hover:bg-[#F4F1EA] hover:text-[#1C1C1E]"
            aria-label="Close"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
          </button>

          {/* Small logo for phones, where the brand panel is hidden */}
          <span className="mb-5 flex h-10 w-10 items-center justify-center rounded-xl bg-[#5E7C5C] text-white md:hidden">
            <LogoMark />
          </span>

          <h2 id="auth-title" className="text-[24px] font-semibold tracking-tight text-[#1C1C1E]">
            {mode === "login" ? "Welcome back" : "Create your account"}
          </h2>

          <div className="mt-6 flex min-h-[44px] justify-center">
            <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID} locale="en">
              <GoogleLogin
                onSuccess={(res) => handleGoogle(res.credential)}
                onError={() => setError("Google sign-in failed. Please try again.")}
                text={mode === "login" ? "continue_with" : "signup_with"}
                shape="pill"
                size="large"
                width="300"
              />
            </GoogleOAuthProvider>
          </div>

          <div className="my-5 flex items-center gap-3 text-[12px] text-[#9A968C]">
            <span className="h-px flex-1 bg-[#ECE7DD]" />
            or
            <span className="h-px flex-1 bg-[#ECE7DD]" />
          </div>

          <form
            className="space-y-4"
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              handleSubmit();
            }}
          >
            <div>
              <label htmlFor="auth-email" className="mb-1.5 block text-[13px] font-medium text-[#1C1C1E]">
                Email
              </label>
              <input
                id="auth-email"
                type="email"
                name="email"
                autoComplete="username"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  setError("");
                }}
                onBlur={() => setEmailTouched(true)}
                aria-invalid={!!emailHint}
                aria-describedby={emailHint ? "auth-email-error" : undefined}
                className={inputClass(!!emailHint)}
              />
              {emailHint && (
                <p id="auth-email-error" className="mt-1.5 flex items-center gap-1.5 text-[12.5px] text-red-600">
                  <AlertIcon /> {emailHint}
                </p>
              )}
            </div>

            <div>
              <label htmlFor="auth-password" className="mb-1.5 block text-[13px] font-medium text-[#1C1C1E]">
                Password
              </label>
              <div className="relative">
                <input
                  id="auth-password"
                  type={showPassword ? "text" : "password"}
                  name="password"
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                  placeholder={mode === "login" ? "Your password" : "Create a password"}
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    setError("");
                  }}
                  onBlur={() => setPasswordTouched(true)}
                  aria-invalid={passwordTooShort}
                  aria-describedby="auth-password-hint"
                  className={`${inputClass(passwordTooShort)} pr-11`}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  onMouseDown={(e) => e.preventDefault()} // keep the cursor in the password box
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  title={showPassword ? "Hide password" : "Show password"}
                  className="absolute right-2 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-lg text-[#9A968C] transition hover:bg-[#F4F1EA] hover:text-[#5E7C5C]"
                >
                  <EyeIcon off={showPassword} />
                </button>
              </div>
              {/* Sign up shows the rule up front; it turns red only if the password is too short */}
              {passwordTooShort ? (
                <p id="auth-password-hint" className="mt-1.5 flex items-center gap-1.5 text-[12.5px] text-red-600">
                  <AlertIcon /> At least {MIN_PASSWORD_LENGTH} characters
                </p>
              ) : signingUp ? (
                <p id="auth-password-hint" className="mt-1.5 text-[12.5px] text-[#9A968C]">
                  Use at least {MIN_PASSWORD_LENGTH} characters.
                </p>
              ) : null}
            </div>

            {error && (
              <div role="alert" className="flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 px-3.5 py-2.5 text-[13px] text-red-700">
                <span className="mt-px"><AlertIcon /></span>
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#5E7C5C] py-3 text-[14.5px] font-semibold text-white shadow-sm transition hover:bg-[#4E684C] active:scale-[0.99] disabled:cursor-wait disabled:opacity-70"
            >
              {loading && (
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.3" strokeWidth="3" />
                  <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                </svg>
              )}
              {loading
                ? mode === "login"
                  ? "Logging in…"
                  : "Creating account…"
                : mode === "login"
                  ? "Log in"
                  : "Create account"}
            </button>
          </form>

          <p className="mt-6 text-center text-[13.5px] text-[#6B6B6B]">
            {mode === "login" ? (
              <>
                New to DocQuery?{" "}
                <button
                  type="button"
                  onClick={() => switchMode("signup")}
                  className="font-semibold text-[#5E7C5C] underline-offset-4 hover:underline"
                >
                  Create an account
                </button>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <button
                  type="button"
                  onClick={() => switchMode("login")}
                  className="font-semibold text-[#5E7C5C] underline-offset-4 hover:underline"
                >
                  Log in
                </button>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
