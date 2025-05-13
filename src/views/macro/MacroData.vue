<template>
  <div class="macro-data-container">
    <a-spin :spinning="loading" tip="数据加载中...">
      <a-card>
        <div class="search-container">
          <a-row :gutter="16" class="search-row" align="middle">
            <a-col :xs="24" :sm="24" :md="6" :lg="6" :xl="6">
              <div class="search-item">
                <div class="input-with-label">
                  <div class="label-container">
                    <BarChartOutlined class="label-icon" />
                    <a-label>宏观数据</a-label>
                  </div>
                  <a-select
                    v-model:value="searchForm.types"
                    placeholder="请选择宏观数据"
                    style="width: 100%"
                    @change="handleTypeChange"
                  >
                    <a-select-option value="CPI">
                      <template #icon><LineChartOutlined /></template>
                      CPI
                    </a-select-option>
                    <a-select-option value="PPI">
                      <template #icon><LineChartOutlined /></template>
                      PPI
                    </a-select-option>
                    <a-select-option value="PMI">
                      <template #icon><LineChartOutlined /></template>
                      PMI
                    </a-select-option>
                  </a-select>
                </div>
              </div>
            </a-col>
            <a-col :xs="24" :sm="24" :md="8" :lg="8" :xl="8">
              <div class="search-item">
                <div class="input-with-label">
                  <div class="label-container">
                    <CalendarOutlined class="label-icon" />
                    <a-label>日期范围</a-label>
                  </div>
                  <a-range-picker
                    v-model:value="dateRange"
                    style="width: 100%"
                    @change="handleDateChange"
                  />
                </div>
              </div>
            </a-col>
            <a-col :xs="24" :sm="24" :md="4" :lg="4" :xl="4">
              <div class="search-item">
                <a-button type="primary" @click="handleSearch">
                  <template #icon><SearchOutlined /></template>
                  查询
                </a-button>
              </div>
            </a-col>
            <a-col :xs="24" :sm="24" :md="6" :lg="6" :xl="6">
              <div class="search-item">
                <a-space>
                  <a-button @click="handleExportCsv">
                    <template #icon><FileExcelOutlined /></template>
                    导出CSV
                  </a-button>
                  <a-button @click="handleExportExcel">
                    <template #icon><FileExcelOutlined /></template>
                    批量导出Excel
                  </a-button>
                </a-space>
              </div>
            </a-col>
          </a-row>
        </div>

        <div class="chart-container">
          <v-chart class="chart" :option="chartOption" autoresize />
        </div>

        <!-- 缩放控制条 -->
        <div class="zoom-control-container">
          <a-row :gutter="16" align="middle">
            <a-col :span="2">
              <span class="zoom-label">数据范围：</span>
            </a-col>
            <a-col :span="20">
              <a-slider
                v-model:value="chartZoom"
                :min="1"
                :max="100"
                :step="1"
                :tooltip-visible="false"
                class="custom-slider"
                @change="handleZoomChange"
              />
            </a-col>
            <a-col :span="2">
              <a-tooltip title="重置缩放">
                <a-button @click="handleZoomReset">
                  <template #icon><UndoOutlined /></template>
                </a-button>
              </a-tooltip>
            </a-col>
          </a-row>
        </div>

        <a-table
          :columns="columns"
          :data-source="tableData"
          :pagination="pagination"
          @change="handleTableChange"
        />
      </a-card>
    </a-spin>
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
import {
  UndoOutlined,
  BarChartOutlined,
  CalendarOutlined,
  SearchOutlined,
  FileExcelOutlined,
  LineChartOutlined,
} from '@ant-design/icons-vue'
import { getMacroDataMacroGetMacroDataPost, getMacroCsvMacroGetMacroCsvPost } from '@/api/macro'
import { batchExportToExcelCommonBatchExportToExcelPost } from '@/api/common'
import type { MacroDataItem } from '@/typings/macro'

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
    sortDirections: ['ascend', 'descend'],
    defaultSortOrder: 'descend',
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

// 缩放栏响应式变量
const chartZoom = ref(50)

// 加载状态
const loading = ref(false)

// 处理日期变化
const handleDateChange = (dates: [Dayjs, Dayjs]) => {
  if (dates) {
    searchForm.value.start_date = dates[0].format('YYYY-MM-DD')
    searchForm.value.end_date = dates[1].format('YYYY-MM-DD')
  }
}

// 处理数据类型变化
const handleTypeChange = () => {
  handleSearch()
}

// 处理缩放变化
const handleZoomChange = (value: number) => {
  const dataLength = tableData.value.length
  let startIndex = 0
  let endIndex = dataLength

  if (value < 50) {
    // 放大左侧数据
    const zoomFactor = (50 - value) / 50 // 0 到 1 的缩放因子
    const visibleCount = Math.floor(dataLength * (1 - zoomFactor))
    endIndex = Math.min(dataLength, startIndex + visibleCount)
  } else if (value > 50) {
    // 放大右侧数据
    const zoomFactor = (value - 50) / 50 // 0 到 1 的缩放因子
    const visibleCount = Math.floor(dataLength * (1 - zoomFactor))
    startIndex = Math.max(0, endIndex - visibleCount)
  }

  // 更新图表数据
  const visibleData = tableData.value.slice(startIndex, endIndex)
  updateChartData(visibleData)
}

// 重置缩放
const handleZoomReset = () => {
  chartZoom.value = 50
  updateChartData(tableData.value)
}

// 更新图表数据
const updateChartData = (data = tableData.value) => {
  // 确保图表数据是按时间升序排列
  const sortedData = [...data].sort(
    (a, b) => dayjs(a.report_date).valueOf() - dayjs(b.report_date).valueOf(),
  )
  const dates = sortedData.map((item) => dayjs(item.report_date).format('YYYY-MM-DD'))
  const currentValues = sortedData.map((item) => item.current_value)
  const forecastValues = sortedData.map((item) => item.forecast_value)
  const previousValues = sortedData.map((item) => item.previous_value)

  chartOption.value.xAxis.data = dates
  chartOption.value.series[0].data = currentValues
  chartOption.value.series[1].data = forecastValues
  chartOption.value.series[2].data = previousValues

  // 根据数据类型调整 Y 轴范围
  if (searchForm.value.types === 'PMI') {
    const min = Math.min(...currentValues, ...forecastValues, ...previousValues)
    const max = Math.max(...currentValues, ...forecastValues, ...previousValues)
    const padding = 2 // 上下各留 2 个单位的空间
    chartOption.value.yAxis.min = Math.floor(min - padding)
    chartOption.value.yAxis.max = Math.ceil(max + padding)
  } else {
    // 其他数据类型使用自动缩放
    chartOption.value.yAxis.min = undefined
    chartOption.value.yAxis.max = undefined
  }
}

// 搜索
const handleSearch = async () => {
  loading.value = true
  try {
    const response = await getMacroDataMacroGetMacroDataPost(searchForm.value)
    if (response.data.code === 200) {
      const data = response.data.data as MacroDataItem[]
      // 按日期升序排序（图表）
      const sortedDataForChart = [...data].sort(
        (a, b) => dayjs(a.report_date).valueOf() - dayjs(b.report_date).valueOf(),
      )
      // 按日期降序排序（表格）
      const sortedDataForTable = [...data].sort(
        (a, b) => dayjs(b.report_date).valueOf() - dayjs(a.report_date).valueOf(),
      )
      tableData.value = sortedDataForTable
      pagination.value.total = sortedDataForTable.length
      // 重置缩放
      chartZoom.value = 50
      updateChartData(sortedDataForChart)
    } else {
      message.error(response.data.msg || '获取数据失败')
    }
  } catch (error) {
    message.error('获取数据失败')
    console.error(error)
  } finally {
    loading.value = false
  }
}

// 导出CSV
const handleExportCsv = async () => {
  loading.value = true
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
  } finally {
    loading.value = false
  }
}

// 导出Excel
const handleExportExcel = async () => {
  loading.value = true
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
  } finally {
    loading.value = false
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
  padding: 16px;
  background: linear-gradient(to right, #fafafa, #f5f5f5);
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.search-row {
  display: flex;
  align-items: center;
}

.search-item {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.input-with-label {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
}

.label-container {
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}

.label-icon {
  font-size: 16px;
  color: #1890ff;
}

/* 响应式布局 */
@media screen and (max-width: 768px) {
  .search-row {
    flex-direction: column;
    gap: 16px;
  }

  .search-item {
    width: 100%;
  }

  .input-with-label {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .label-container {
    margin-bottom: 4px;
  }

  :deep(.ant-space) {
    width: 100%;
    display: flex;
    justify-content: space-between;
  }

  :deep(.ant-btn) {
    flex: 1;
  }
}

:deep(.ant-select-selector) {
  border-radius: 8px !important;
  transition: all 0.3s ease !important;
}

:deep(.ant-select-selector:hover) {
  border-color: #40a9ff !important;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.1) !important;
}

:deep(.ant-select-focused .ant-select-selector) {
  border-color: #40a9ff !important;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2) !important;
}

:deep(.ant-picker) {
  border-radius: 8px !important;
  transition: all 0.3s ease !important;
}

:deep(.ant-picker:hover) {
  border-color: #40a9ff !important;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.1) !important;
}

:deep(.ant-picker-focused) {
  border-color: #40a9ff !important;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2) !important;
}

:deep(.ant-btn) {
  border-radius: 8px !important;
  transition: all 0.3s ease !important;
}

:deep(.ant-btn:hover) {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

:deep(.ant-btn:active) {
  transform: translateY(0);
}

.chart-container {
  height: 400px;
  margin-bottom: 24px;
}

.chart {
  height: 100%;
  width: 100%;
}

.zoom-control-container {
  margin: 16px 0;
  padding: 16px;
  background: linear-gradient(to right, #fafafa, #f5f5f5);
  border-radius: 12px;
  position: relative;
  z-index: 1;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.zoom-label {
  font-size: 14px;
  color: #666;
  font-weight: 500;
}

:deep(.custom-slider) {
  margin: 10px 0;
  cursor: pointer;
}

:deep(.custom-slider .ant-slider-rail) {
  height: 6px;
  background: linear-gradient(to right, #e6f7ff, #bae7ff);
  border-radius: 3px;
  transition: all 0.3s ease;
}

:deep(.custom-slider .ant-slider-track) {
  height: 6px;
  background: linear-gradient(to right, #1890ff, #40a9ff);
  border-radius: 3px;
  transition: all 0.3s ease;
}

:deep(.custom-slider .ant-slider-handle) {
  width: 20px;
  height: 20px;
  margin-top: -7px;
  background: #fff;
  border: 2px solid #1890ff;
  box-shadow: 0 2px 6px rgba(24, 144, 255, 0.2);
  transition: all 0.3s ease;
  cursor: grab;
}

:deep(.custom-slider .ant-slider-handle:active) {
  cursor: grabbing;
  border-color: #096dd9;
  transform: scale(0.95);
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.3);
}

:deep(.custom-slider:hover .ant-slider-rail) {
  background: linear-gradient(to right, #e6f7ff, #91d5ff);
}

:deep(.custom-slider:hover .ant-slider-track) {
  background: linear-gradient(to right, #40a9ff, #69c0ff);
}

:deep(.custom-slider:hover .ant-slider-handle) {
  border-color: #40a9ff;
  transform: scale(1.1);
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.3);
}

:deep(.custom-slider .ant-slider-handle::after) {
  display: none;
}

:deep(.custom-slider .ant-slider-handle::before) {
  display: none;
}

:deep(.custom-slider .ant-slider-mark) {
  top: 14px;
}

:deep(.custom-slider .ant-slider-mark-text) {
  color: #999;
  font-size: 12px;
  transform: translateX(-50%);
}

:deep(.custom-slider .ant-slider-mark-text-active) {
  color: #666;
}

:deep(.custom-slider .ant-slider-handle:focus) {
  box-shadow: 0 0 0 3px rgba(24, 144, 255, 0.1);
}
</style>
