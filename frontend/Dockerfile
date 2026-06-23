# aiforge frontend — build the Vite SPA, serve it with nginx, and proxy /api
# and /health to the backend service.
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
# (package-lock.json is not committed in this lightweight repo; `npm install`
# resolves the lockfile at build time.)
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS serve
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
