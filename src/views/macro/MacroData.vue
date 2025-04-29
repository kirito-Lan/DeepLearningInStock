<template>
  <div class="macro-data-container">
    <a-card>
      <div class="search-container">
        <a-row :gutter="16" class="search-row" justify="center">
          <a-col :span="6">
            <a-select
              v-model:value="searchForm.types"
              placeholder="请选择数据类型"
              style="width: 100%"
            >
              <a-select-option value="CPI">CPI</a-select-option>
              <a-select-option value="PPI">PPI</a-select-option>
              <a-select-option value="GDP">GDP</a-select-option>
            </a-select>
          </a-col>
          <a-col :span="8">
            <a-range-picker
              v-model:value="dateRange"
              style="width: 100%"
              @change="handleDateChange"
            />
          </a-col>
          <a-col :span="4">
            <a-button type="primary" @click="handleSearch">查询</a-button>
          </a-col>
          <a-col :span="6">
            <a-space>
              <a-button @click="handleExportCsv">导出CSV</a-button>
              <a-button @click="handleExportExcel">导出Excel</a-button>
            </a-space>
          </a-col>
        </a-row>
      </div>

      <div class="chart-container">
        <v-chart class="chart" :option="chartOption" autoresize />
      </div>

      <a-table
        :columns="columns"
        :data-source="tableData"
        :pagination="pagination"
        @change="handleTableChange"
      />
    </a-card>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import { message } from 'ant-design-vue'
import dayjs from 'dayjs'
import type { Dayjs } from 'dayjs'
import { getMacroDataMacroGetMacroDataPost, getMacroCsvMacroGetMacroCsvPost } from '@/api/macro'
import { batchExportToExcelCommonBatchExportToExcelPost } from '@/api/common'
import type { MacroDataItem } from '@/typings/macro'
import type { BaseResponse } from '@/typings/api'

// 注册必要的组件
use([CanvasRenderer, LineChart, TitleComponent, TooltipComponent, LegendComponent, GridComponent])

// 搜索表单
const searchForm = ref({
  types: 'CPI',
  start_date: '',
  end_date: '',
})

const dateRange = ref<[Dayjs, Dayjs]>([dayjs().subtract(1, 'year'), dayjs()])

// 图表配置
const chartOption = ref({
  tooltip: {
    trigger: 'axis',
  },
  legend: {
    data: ['当前值', '预测值', '前值'],
  },
  xAxis: {
    type: 'category',
    data: [] as string[],
  },
  yAxis: {
    type: 'value',
  },
  series: [
    {
      name: '当前值',
      type: 'line',
      data: [] as number[],
    },
    {
      name: '预测值',
      type: 'line',
      data: [] as number[],
    },
    {
      name: '前值',
      type: 'line',
      data: [] as number[],
    },
  ],
})

// 表格配置
const columns = [
  {
    title: '发布日期',
    dataIndex: 'report_date',
    key: 'report_date',
    sorter: true,
    customRender: ({ text }: { text: string }) => {
      return dayjs(text).format('YYYY-MM-DD')
    },
  },
  {
    title: '当前值',
    dataIndex: 'current_value',
    key: 'current_value',
  },
  {
    title: '预测值',
    dataIndex: 'forecast_value',
    key: 'forecast_value',
  },
  {
    title: '前值',
    dataIndex: 'previous_value',
    key: 'previous_value',
  },
]

const tableData = ref<MacroDataItem[]>([])
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
})

// 处理日期变化
const handleDateChange = (dates: [Dayjs, Dayjs]) => {
  if (dates) {
    searchForm.value.start_date = dates[0].format('YYYY-MM-DD')
    searchForm.value.end_date = dates[1].format('YYYY-MM-DD')
  }
}

// 搜索
const handleSearch = async () => {
  try {
    const response = await getMacroDataMacroGetMacroDataPost(searchForm.value)
    if (response.data.code === 200) {
      const data = response.data.data as MacroDataItem[]
      tableData.value = data
      pagination.value.total = data.length

      // 更新图表数据
      const dates = data.map((item) => dayjs(item.report_date).format('YYYY-MM-DD'))
      const currentValues = data.map((item) => item.current_value)
      const forecastValues = data.map((item) => item.forecast_value)
      const previousValues = data.map((item) => item.previous_value)

      chartOption.value.xAxis.data = dates
      chartOption.value.series[0].data = currentValues
      chartOption.value.series[1].data = forecastValues
      chartOption.value.series[2].data = previousValues
    } else {
      message.error(response.data.msg || '获取数据失败')
    }
  } catch (error) {
    message.error('获取数据失败')
    console.error(error)
  }
}

// 导出CSV
const handleExportCsv = async () => {
  try {
    const response = await getMacroCsvMacroGetMacroCsvPost(searchForm.value)
    if (response.data) {
      const blob = new Blob([response.data], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `macro_data_${searchForm.value.types}_${dayjs().format('YYYY-MM-DD')}.csv`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    }
  } catch (error) {
    message.error('导出CSV失败')
    console.error(error)
  }
}

// 导出Excel
const handleExportExcel = async () => {
  try {
    const response = await batchExportToExcelCommonBatchExportToExcelPost(
      {
        export_type: 'macro',
        start_date: searchForm.value.start_date,
        end_date: searchForm.value.end_date,
      },
      {
        responseType: 'blob',
      },
    )

    if (response.data) {
      const blob = new Blob([response.data], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `macro_data_${dayjs().format('YYYY-MM-DD')}.xlsx`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    }
  } catch (error) {
    message.error('导出Excel失败')
    console.error(error)
  }
}

// 处理表格变化
const handleTableChange = (pag: { current: number; pageSize: number }) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
}

onMounted(() => {
  handleDateChange(dateRange.value)
  handleSearch()
})
</script>

<style scoped>
.macro-data-container {
  padding: 24px;
}

.search-container {
  margin-bottom: 24px;
}

.search-row {
  display: flex;
  align-items: center;
}

.chart-container {
  height: 400px;
  margin-bottom: 24px;
}

.chart {
  height: 100%;
  width: 100%;
}
</style>
