import { createRouter, createWebHistory } from 'vue-router'
import TheWelcome from '@/components/TheWelcome.vue'


const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: TheWelcome,
    },

  ],
})

export default router
