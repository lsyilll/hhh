import {defineConfig} from 'vite';import react from '@vitejs/plugin-react';
export default defineConfig({base:'/platform/',plugins:[react()],server:{host:'127.0.0.1',proxy:{'/api':'http://127.0.0.1:7860'}},build:{chunkSizeWarningLimit:1600,rollupOptions:{output:{manualChunks:{react:['react','react-dom'],charts:['echarts'],antd:['antd','@ant-design/icons']}}}}});
