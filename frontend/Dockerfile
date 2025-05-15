# Stage 1 - Build
FROM node:20-alpine AS build

# Set the working directory
WORKDIR /app

# Install dependencies separately to leverage Docker layer caching
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci --frozen-lockfile

# Copy the rest of the application code
COPY . .

# Build the application
RUN npm run build

# Stage 2 - Production
FROM node:20-alpine AS production

# Set the working directory
WORKDIR /app

# Install a lightweight HTTP server
RUN npm install -g serve

# Copy the build output from the build image
COPY --from=build /app/dist ./dist

# Expose the port the app runs on
EXPOSE 5173

# Command to serve the built files
CMD ["serve", "-s", "dist", "-l", "5173"]