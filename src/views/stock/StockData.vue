<template>
  <div class="stock-data-container">
    <a-card>
      <div class="search-container">
        <a-row :gutter="16" class="search-row" justify="center">
          <a-col :span="2">
            <div class="label-container">
              <StockOutlined class="label-icon" />
              <a-label>选择股指</a-label>
            </div>
          </a-col>
          <a-col :span="6">
            <a-select
              v-model:value="searchForm.stock_code"
              placeholder="请选择股票"
              style="width: 100%"
              @change="handleStockChange"
            >
              <a-select-option v-for="stock in stockList" :key="stock.code" :value="stock.code">
                {{ stock.name }}
              </a-select-option>
            </a-select>
          </a-col>
          <a-col :span="2">
            <div class="label-container">
              <CalendarOutlined class="label-icon" />
              <a-label>日期范围</a-label>
            </div>
          </a-col>
          <a-col :span="8">
            <a-range-picker
              v-model:value="dateRange"
              style="width: 100%"
              :disabledDate="disabledDate"
              @change="handleDateChange"
            />
          </a-col>
          <a-col :span="6">
            <a-space>
              <a-button type="primary" @click="handleSearch">
                <template #icon><SearchOutlined /></template>
                查询
              </a-button>
              <a-button @click="handleExportCsv">
                <template #icon><FileExcelOutlined /></template>
                导出CSV
              </a-button>
              <a-button @click="handleExportExcel">
                <template #icon><FileExcelOutlined /></template>
                导出Excel
              </a-button>
            </a-space>
          </a-col>
        </a-row>
      </div>

      <!-- 最新股票资讯面板 -->
      <div v-if="tableData.length > 0" class="latest-stock-info">
        <a-card>
          <template #title>
            <div class="latest-stock-title">
              <InfoCircleOutlined />
              <span>最新股票资讯</span>
              <a-tag color="blue" class="time-tag">
                <ClockCircleOutlined />
                {{ tableData[0].trade_date }}
              </a-tag>
            </div>
          </template>
          <a-row :gutter="16">
            <a-col :span="4">
              <div class="info-item">
                <div class="info-label"><RiseOutlined /> 开盘价</div>
                <div class="info-value">{{ tableData[0].open_price }}</div>
              </div>
            </a-col>
            <a-col :span="4">
              <div class="info-item">
                <div class="info-label"><FallOutlined /> 收盘价</div>
                <div class="info-value">{{ tableData[0].close_price }}</div>
              </div>
            </a-col>
            <a-col :span="4">
              <div class="info-item">
                <div class="info-label"><ArrowUpOutlined /> 最高价</div>
                <div class="info-value">{{ tableData[0].high_price }}</div>
              </div>
            </a-col>
            <a-col :span="4">
              <div class="info-item">
                <div class="info-label"><ArrowDownOutlined /> 最低价</div>
                <div class="info-value">{{ tableData[0].low_price }}</div>
              </div>
            </a-col>
            <a-col :span="4">
              <div class="info-item">
                <div class="info-label"><BarChartOutlined /> 成交量</div>
                <div class="info-value">{{ tableData[0].volume }}</div>
              </div>
            </a-col>
            <a-col :span="4">
              <div class="info-item">
                <div class="info-label"><LineChartOutlined /> 涨跌幅</div>
                <div class="info-value" :class="getChangeRateClass(tableData[0].change_rate)">
                  {{ tableData[0].change_rate }}%
                </div>
              </div>
            </a-col>
          </a-row>
        </a-card>
      </div>

      <div class="chart-container">
        <div class="chart-header">
          <a-radio-group v-model:value="chartType" button-style="solid" class="chart-type-selector">
            <a-radio-button value="price">价格走势</a-radio-button>
            <a-radio-button value="volume">成交量</a-radio-button>
            <a-radio-button value="turnover">成交额</a-radio-button>
            <a-radio-button value="amplitude">振幅</a-radio-button>
            <a-radio-button value="change">涨跌幅</a-radio-button>
          </a-radio-group>
        </div>
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
              :tooltip-visible="true"
              class="custom-slider"
              @change="handleZoomChange"
            />
          </a-col>
          <a-col :span="2">
            <a-button-group>
              <a-tooltip title="重置缩放">
                <a-button @click="handleZoomReset">
                  <template #icon><UndoOutlined /></template>
                </a-button>
              </a-tooltip>
              <a-tooltip title="放大">
                <a-button @click="handleZoomIn">
                  <template #icon><ZoomInOutlined /></template>
                </a-button>
              </a-tooltip>
              <a-tooltip title="缩小">
                <a-button @click="handleZoomOut">
                  <template #icon><ZoomOutOutlined /></template>
                </a-button>
              </a-tooltip>
            </a-button-group>
          </a-col>
        </a-row>
      </div>

      <a-table
        :columns="columns"
        :data-source="tableData"
        :pagination="pagination"
        @change="handleTableChange"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'action'">
            <a @click="showDetail(record)">查看详情</a>
          </template>
        </template>
      </a-table>
    </a-card>

    <a-modal v-model:visible="detailVisible" :footer="null" width="800px">
      <div class="detail-header">
        <StockOutlined class="detail-icon" />
        <span class="detail-title">股票详情</span>
      </div>
      <a-descriptions :column="2" bordered>
        <a-descriptions-item label="交易日期">
          <CalendarOutlined /> {{ detailData.trade_date }}
        </a-descriptions-item>
        <a-descriptions-item label="开盘价">
          <RiseOutlined /> {{ detailData.open_price }}
        </a-descriptions-item>
        <a-descriptions-item label="收盘价">
          <FallOutlined /> {{ detailData.close_price }}
        </a-descriptions-item>
        <a-descriptions-item label="最高价">
          <ArrowUpOutlined /> {{ detailData.high_price }}
        </a-descriptions-item>
        <a-descriptions-item label="最低价">
          <ArrowDownOutlined /> {{ detailData.low_price }}
        </a-descriptions-item>
        <a-descriptions-item label="成交量">
          <BarChartOutlined /> {{ detailData.volume }}
        </a-descriptions-item>
        <a-descriptions-item label="成交额">
          <MoneyCollectOutlined /> {{ detailData.turnover_amount }}
        </a-descriptions-item>
        <a-descriptions-item label="振幅">
          <DashboardOutlined /> {{ detailData.amplitude }}%
        </a-descriptions-item>
        <a-descriptions-item label="涨跌幅">
          <LineChartOutlined /> {{ detailData.change_rate }}%
        </a-descriptions-item>
        <a-descriptions-item label="涨跌额">
          <FundOutlined /> {{ detailData.change_amount }}
        </a-descriptions-item>
        <a-descriptions-item label="换手率">
          <SwapOutlined /> {{ detailData.turnover_rate }}%
        </a-descriptions-item>
      </a-descriptions>
    </a-modal>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, BarChart } from 'echarts/charts'
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
  StockOutlined,
  CalendarOutlined,
  SearchOutlined,
  FileExcelOutlined,
  InfoCircleOutlined,
  ClockCircleOutlined,
  RiseOutlined,
  FallOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  BarChartOutlined,
  LineChartOutlined,
  MoneyCollectOutlined,
  DashboardOutlined,
  FundOutlined,
  SwapOutlined,
  UndoOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
} from '@ant-design/icons-vue'
import { getStockListStockGetStockListGet } from '@/api/stock'
import { getStockDataStockGetStockDataPost } from '@/api/stock'
import { getStockCsvStockGetStockCsvPost } from '@/api/stock'
import { batchExportToExcelCommonBatchExportToExcelPost } from '@/api/common'
import type { StockDataItem } from '@/typings/stock'

// 注册必要的组件
use([
  CanvasRenderer,
  LineChart,
  BarChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
])

// 搜索表单
const searchForm = ref({
  stock_code: '',
  start_date: '',
  end_date: '',
})

const dateRange = ref<[Dayjs, Dayjs]>([dayjs().subtract(1, 'year'), dayjs()])
const stockList = ref<{ name: string; code: string }[]>([])
const chartType = ref('price')

// 禁用日期选择
const disabledDate = (current: Dayjs) => {
  return current && current > dayjs().endOf('day')
}

// 图表配置
const chartOption = ref({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross',
      label: {
        backgroundColor: '#6a7985',
      },
    },
    formatter: (params: any) => {
      const date = params[0].axisValue
      const data = tableData.value.find((item) => item.trade_date === date)
      if (!data) return ''

      let result = `<div style="font-weight: bold">${date}</div>`

      // 根据当前图表类型显示不同的数据
      switch (chartType.value) {
        case 'volume':
          result += `<div>成交量：${data.volume}</div>`
          break
        case 'turnover':
          result += `<div>成交额：${data.turnover_amount}</div>`
          break
        case 'amplitude':
          result += `<div>振幅：${data.amplitude}%</div>`
          break
        case 'change':
          result += `<div>涨跌幅：${data.change_rate}%</div>`
          break
        default:
          params.forEach((param: any) => {
            const color = param.color
            result += `<div style="color: ${color}">${param.seriesName}：${param.value}</div>`
          })
      }
      return result
    },
  },
  legend: {
    data: [] as string[],
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '10%',
    containLabel: true,
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    data: [] as string[],
  },
  yAxis: {
    type: 'value',
    scale: true,
  },
  series: [] as any[],
})

// 表格配置
const columns = [
  {
    title: '交易日期',
    dataIndex: 'trade_date',
    key: 'trade_date',
    sorter: true,
    sortDirections: ['ascend', 'descend'],
    defaultSortOrder: 'descend',
  },
  {
    title: '开盘价',
    dataIndex: 'open_price',
    key: 'open_price',
    sorter: true,
  },
  {
    title: '收盘价',
    dataIndex: 'close_price',
    key: 'close_price',
    sorter: true,
  },
  {
    title: '最高价',
    dataIndex: 'high_price',
    key: 'high_price',
    sorter: true,
  },
  {
    title: '最低价',
    dataIndex: 'low_price',
    key: 'low_price',
    sorter: true,
  },
  {
    title: '操作',
    key: 'action',
  },
]

const tableData = ref<StockDataItem[]>([])
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
})

// 详情弹窗
const detailVisible = ref(false)
const detailData = ref<StockDataItem>({} as StockDataItem)

// 添加新的响应式变量
const chartZoom = ref(50)

// 获取股票列表
const getStockList = async () => {
  try {
    const response = await getStockListStockGetStockListGet()
    if (response.data.code === 200) {
      stockList.value = response.data.data
      // 自动选择第一个股票并加载数据
      if (stockList.value.length > 0) {
        searchForm.value.stock_code = stockList.value[0].code
        handleSearch()
      }
    }
  } catch (error) {
    message.error('获取股票列表失败')
    console.error(error)
  }
}

// 处理日期变化
const handleDateChange = (dates: [Dayjs, Dayjs]) => {
  if (dates) {
    searchForm.value.start_date = dates[0].format('YYYY-MM-DD')
    searchForm.value.end_date = dates[1].format('YYYY-MM-DD')
  }
}

// 处理股票选择变化
const handleStockChange = () => {
  handleSearch()
}

// 处理缩放变化
const handleZoomChange = (value: number) => {
  // 根据缩放值调整图表显示范围
  const dataLength = tableData.value.length
  const visibleCount = Math.floor(dataLength * (value / 100))
  const startIndex = Math.max(0, dataLength - visibleCount)
  const visibleData = tableData.value.slice(startIndex)

  // 更新图表数据
  updateChartData(visibleData)
}

// 重置缩放
const handleZoomReset = () => {
  chartZoom.value = 50
  handleZoomChange(50)
}

// 放大
const handleZoomIn = () => {
  chartZoom.value = Math.min(100, chartZoom.value + 10)
  handleZoomChange(chartZoom.value)
}

// 缩小
const handleZoomOut = () => {
  chartZoom.value = Math.max(1, chartZoom.value - 10)
  handleZoomChange(chartZoom.value)
}

// 获取涨跌幅的样式类
const getChangeRateClass = (rate: number) => {
  if (rate < 0) return 'positive-change'
  if (rate > 0) return 'negative-change'
  return ''
}

// 更新图表数据
const updateChartData = (data = tableData.value) => {
  // 反转数据顺序，使最新的数据在右边
  const reversedData = [...data].reverse()
  const dates = reversedData.map((item) => item.trade_date)
  chartOption.value.xAxis.data = dates

  switch (chartType.value) {
    case 'price':
      chartOption.value.legend.data = ['开盘价', '收盘价', '最高价', '最低价']
      chartOption.value.series = [
        {
          name: '开盘价',
          type: 'line',
          data: reversedData.map((item) => parseFloat(item.open_price)),
          emphasis: {
            focus: 'series',
          },
          itemStyle: {
            color: '#1890ff',
          },
        },
        {
          name: '收盘价',
          type: 'line',
          data: reversedData.map((item) => parseFloat(item.close_price)),
          emphasis: {
            focus: 'series',
          },
          itemStyle: {
            color: '#52c41a',
          },
        },
        {
          name: '最高价',
          type: 'line',
          data: reversedData.map((item) => parseFloat(item.high_price)),
          emphasis: {
            focus: 'series',
          },
          itemStyle: {
            color: '#f5222d',
          },
        },
        {
          name: '最低价',
          type: 'line',
          data: reversedData.map((item) => parseFloat(item.low_price)),
          emphasis: {
            focus: 'series',
          },
          itemStyle: {
            color: '#fa8c16',
          },
        },
      ]
      break
    case 'volume':
      chartOption.value.legend.data = ['成交量']
      chartOption.value.series = [
        {
          name: '成交量',
          type: 'line',
          data: reversedData.map((item) => item.volume),
          emphasis: {
            focus: 'series',
          },
          itemStyle: {
            color: '#1890ff',
          },
        },
      ]
      break
    case 'turnover':
      chartOption.value.legend.data = ['成交额']
      chartOption.value.series = [
        {
          name: '成交额',
          type: 'line',
          data: reversedData.map((item) => parseFloat(item.turnover_amount)),
          emphasis: {
            focus: 'series',
          },
          itemStyle: {
            color: '#1890ff',
          },
        },
      ]
      break
    case 'amplitude':
      chartOption.value.legend.data = ['振幅']
      chartOption.value.series = [
        {
          name: '振幅',
          type: 'line',
          data: reversedData.map((item) => parseFloat(item.amplitude)),
          emphasis: {
            focus: 'series',
          },
          itemStyle: {
            color: '#1890ff',
          },
        },
      ]
      break
    case 'change':
      chartOption.value.legend.data = ['涨跌幅']
      chartOption.value.series = [
        {
          name: '涨跌幅',
          type: 'line',
          data: reversedData.map((item) => parseFloat(item.change_rate)),
          emphasis: {
            focus: 'series',
          },
          itemStyle: {
            color: '#1890ff',
          },
        },
      ]
      break
  }
}

// 搜索
const handleSearch = async () => {
  try {
    const response = await getStockDataStockGetStockDataPost(searchForm.value)
    if (response.data.code === 200) {
      tableData.value = response.data.data
      pagination.value.total = tableData.value.length
      updateChartData()
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
    const response = await getStockCsvStockGetStockCsvPost(searchForm.value)
    if (response.data) {
      const blob = new Blob([response.data], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `stock_data_${searchForm.value.stock_code}_${dayjs().format('YYYY-MM-DD')}.csv`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    }
  } catch (error) {
    message.error('导出失败')
    console.error(error)
  }
}

// 导出Excel
const handleExportExcel = async () => {
  try {
    const response = await batchExportToExcelCommonBatchExportToExcelPost(
      {
        export_type: 'stock',
        stock_code: searchForm.value.stock_code,
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
      link.download = `stock_data_${dayjs().format('YYYY-MM-DD')}.xlsx`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    }
  } catch (error) {
    message.error('导出失败')
    console.error(error)
  }
}

// 处理表格变化
const handleTableChange = (
  pag: { current: number; pageSize: number },
  filters: any,
  sorter: any,
) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize

  // 处理排序
  if (sorter && sorter.field) {
    const sortData = [...tableData.value].sort((a, b) => {
      let compareA, compareB

      // 根据字段类型进行不同的排序处理
      if (sorter.field === 'trade_date') {
        compareA = dayjs(a[sorter.field]).valueOf()
        compareB = dayjs(b[sorter.field]).valueOf()
      } else {
        compareA = parseFloat(a[sorter.field])
        compareB = parseFloat(b[sorter.field])
      }

      if (sorter.order === 'ascend') {
        return compareA - compareB
      } else if (sorter.order === 'descend') {
        return compareB - compareA
      }
      return 0
    })

    tableData.value = sortData
  }
}

// 显示详情
const showDetail = (record: StockDataItem) => {
  detailData.value = record
  detailVisible.value = true
}

// 监听图表类型变化
watch(chartType, () => {
  // 重置缩放
  chartZoom.value = 50
  // 更新图表数据
  updateChartData()
})

onMounted(() => {
  handleDateChange(dateRange.value)
  getStockList() // 获取股票列表，会自动选择第一个并加载数据
})
</script>

<style scoped>
.stock-data-container {
  padding: 24px;
}

.search-container {
  margin-bottom: 24px;
}

.search-row {
  display: flex;
  align-items: center;
}

.label-container {
  display: flex;
  align-items: center;
  gap: 8px;
}

.label-icon {
  font-size: 16px;
  color: #1890ff;
}

.latest-stock-info {
  margin-bottom: 24px;
}

.latest-stock-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: bold;
}

.info-item {
  text-align: center;
  padding: 8px;
  border-radius: 4px;
  background-color: #f5f5f5;
}

.info-label {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #666;
  margin-bottom: 4px;
}

.info-value {
  font-size: 16px;
  font-weight: bold;
  color: #333;
}

.time-tag {
  margin-left: 16px;
}

.positive-change {
  color: #52c41a;
}

.negative-change {
  color: #f5222d;
}

.chart-container {
  height: 400px;
  margin-bottom: 24px;
  position: relative;
}

.chart {
  height: 100%;
  width: 100%;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.chart-controls {
  display: flex;
  align-items: center;
  gap: 16px;
}

.detail-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}

.detail-icon {
  font-size: 24px;
  color: #1890ff;
}

.detail-title {
  font-size: 18px;
  font-weight: bold;
}

.zoom-control-container {
  margin: 16px 0;
  padding: 16px;
  background-color: #f5f5f5;
  border-radius: 4px;
  position: relative;
  z-index: 1;
}

.zoom-label {
  font-size: 14px;
  color: #666;
}

:deep(.custom-slider) {
  margin: 10px 0;
}

:deep(.custom-slider .ant-slider-rail) {
  height: 2px;
  background-color: #e8e8e8;
}

:deep(.custom-slider .ant-slider-track) {
  height: 2px;
  background-color: #1890ff;
}

:deep(.custom-slider .ant-slider-handle) {
  width: 12px;
  height: 12px;
  margin-top: -5px;
  background-color: #fff;
  border: 2px solid #1890ff;
  box-shadow: none;
}

:deep(.custom-slider .ant-slider-handle:focus) {
  box-shadow: none;
}

:deep(.custom-slider .ant-slider-handle:hover) {
  border-color: #40a9ff;
}

:deep(.custom-slider .ant-slider-handle:active) {
  border-color: #096dd9;
}

:deep(.custom-slider .ant-slider-tooltip) {
  z-index: 2;
}

:deep(.custom-slider .ant-slider-mark-text) {
  color: #999;
  font-size: 12px;
}

:deep(.custom-slider .ant-slider-mark) {
  top: 10px;
}

:deep(.custom-slider .ant-slider-mark-text-active) {
  color: #666;
}

:deep(.echarts-tooltip) {
  z-index: 1000;
}
</style>
