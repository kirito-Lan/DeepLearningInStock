import { createRouter, createWebHistory } from 'vue-router'
import MacroData from '@/views/macro/MacroData.vue'
import PredictVue from '@/views/Predict/Predict.vue'
import StockDataVue from '../views/stock/StockData.vue'
import EDAVue from '@/views/EDA/eda.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: MacroData,
    },
    {
      path: '/exponent',
      name: 'exponent',
      component: StockDataVue,
    },
    {
      path: '/eda',
      name: 'eda',
      component: EDAVue,
    },
    {
      path: '/predict',
      name: 'predict',
      component: PredictVue,
    },
  ],
})

export default router
