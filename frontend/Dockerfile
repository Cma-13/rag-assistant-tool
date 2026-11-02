# Frontend: Next.js
# --- development stage (used by docker-compose.dev.yml): runs "npm run dev" ---
FROM node:22-alpine AS dev
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0"]

# --- build stage: makes the production build ---
FROM dev AS build
# Where the browser finds the backend (it is baked into the build)
ARG NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

# --- final stage: only what is needed to run ---
FROM node:22-alpine AS prod
WORKDIR /app
ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0
COPY --from=build /app ./
EXPOSE 3000
CMD ["npm", "run", "start"]
