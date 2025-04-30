import { createRouter, createWebHistory } from 'vue-router'
import MacroData from '@/views/macro/MacroData.vue'
import StockDataVue from '../views/stock/StockData.vue'
import TheWelcome from '@/components/TheWelcome.vue'

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
      path: '/predict',
      name: 'predict',
      component: TheWelcome
    },
  ],
})

export default router
