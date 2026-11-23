import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    const localApiOrigin = process.env.STOREPULSE_LOCAL_API_ORIGIN;
    return localApiOrigin
      ? [{ source: "/api/v1/:path*", destination: `${localApiOrigin}/api/v1/:path*` }]
      : [];
  },
};

export default nextConfig;
