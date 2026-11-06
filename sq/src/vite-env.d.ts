/// <reference types="vite/client" />

declare module '*.wgsl?raw' {
  const shaderCode: string;
  export default shaderCode;
}

declare module '*.wgsl' {
  const shaderCode: string;
  export default shaderCode;
}
