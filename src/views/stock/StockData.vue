<template>
  <div class="stock-data-container">
    <a-spin :spinning="loading" tip="数据加载中...">
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
            <a-radio-group
              v-model:value="chartType"
              button-style="solid"
              class="chart-type-selector"
            >
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
        >
          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'action'">
              <a @click="showDetail(record)">查看详情</a>
            </template>
          </template>
        </a-table>
      </a-card>
    </a-spin>

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
import { onMounted, ref, watch, computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import { message } from 'ant-design-vue'
import type { Dayjs } from 'dayjs'
import dayjs from 'dayjs'
import {
  ArrowDownOutlined,
  ArrowUpOutlined,
  BarChartOutlined,
  CalendarOutlined,
  ClockCircleOutlined,
  DashboardOutlined,
  FallOutlined,
  FileExcelOutlined,
  FundOutlined,
  InfoCircleOutlined,
  LineChartOutlined,
  MoneyCollectOutlined,
  RiseOutlined,
  SearchOutlined,
  StockOutlined,
  SwapOutlined,
  UndoOutlined,
} from '@ant-design/icons-vue'
import {
  getStockCsvStockGetStockCsvPost,
  getStockDataStockGetStockDataPost,
  getStockListStockGetStockListGet,
} from '@/api/stock'
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
const priceChartOption = ref({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross',
      label: {
        backgroundColor: '#6a7985',
      },
    },
  },
  legend: {
    data: ['开盘价', '收盘价', '最高价', '最低价'],
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

const volumeChartOption = ref({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross',
      label: {
        backgroundColor: '#6a7985',
      },
    },
  },
  legend: {
    data: ['成交量'],
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

const turnoverChartOption = ref({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross',
      label: {
        backgroundColor: '#6a7985',
      },
    },
  },
  legend: {
    data: ['成交额'],
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

const amplitudeChartOption = ref({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross',
      label: {
        backgroundColor: '#6a7985',
      },
    },
  },
  legend: {
    data: ['振幅'],
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

const changeChartOption = ref({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross',
      label: {
        backgroundColor: '#6a7985',
      },
    },
  },
  legend: {
    data: ['涨跌幅'],
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

// 计算属性：根据当前图表类型返回对应的配置
const chartOption = computed(() => {
  switch (chartType.value) {
    case 'price':
      return priceChartOption.value
    case 'volume':
      return volumeChartOption.value
    case 'turnover':
      return turnoverChartOption.value
    case 'amplitude':
      return amplitudeChartOption.value
    case 'change':
      return changeChartOption.value
    default:
      return priceChartOption.value
  }
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

// 添加加载状态
const loading = ref(false)

// 获取股票列表
const getStockList = async () => {
  loading.value = true
  try {
    const response = await getStockListStockGetStockListGet()
    if (response.data.code === 200) {
      stockList.value = response.data.data
      // 自动选择第一个股票并加载数据
      if (stockList.value.length > 0) {
        searchForm.value.stock_code = stockList.value[0].code
        await handleSearch()
      }
    }
  } catch (error) {
    message.error('获取股票列表失败')
    console.error(error)
  } finally {
    loading.value = false
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
  const dataLength = tableData.value.length
  let startIndex = 0
  let endIndex = dataLength

  if (value < 50) {
    // 放大右侧数据
    const zoomFactor = (50 - value) / 50 // 0 到 1 的缩放因子
    const visibleCount = Math.floor(dataLength * (1 - zoomFactor))
    startIndex = Math.max(0, endIndex - visibleCount)
  } else if (value > 50) {
    // 放大左侧数据
    const zoomFactor = (value - 50) / 50 // 0 到 1 的缩放因子
    const visibleCount = Math.floor(dataLength * (1 - zoomFactor))
    endIndex = Math.min(dataLength, startIndex + visibleCount)
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

  // 更新所有图表的 X 轴数据
  priceChartOption.value.xAxis.data = dates
  volumeChartOption.value.xAxis.data = dates
  turnoverChartOption.value.xAxis.data = dates
  amplitudeChartOption.value.xAxis.data = dates
  changeChartOption.value.xAxis.data = dates

  // 更新价格图表数据
  priceChartOption.value.series = [
    {
      name: '开盘价',
      type: 'line',
      data: reversedData.map((item) => parseFloat(item.open_price)),
      emphasis: { focus: 'series' },
      itemStyle: { color: '#1890ff' },
      symbol: 'none',
    },
    {
      name: '收盘价',
      type: 'line',
      data: reversedData.map((item) => parseFloat(item.close_price)),
      emphasis: { focus: 'series' },
      itemStyle: { color: '#52c41a' },
      symbol: 'none',
    },
    {
      name: '最高价',
      type: 'line',
      data: reversedData.map((item) => parseFloat(item.high_price)),
      emphasis: { focus: 'series' },
      itemStyle: { color: '#f5222d' },
      symbol: 'none',
    },
    {
      name: '最低价',
      type: 'line',
      data: reversedData.map((item) => parseFloat(item.low_price)),
      emphasis: { focus: 'series' },
      itemStyle: { color: '#fa8c16' },
      symbol: 'none',
    },
  ]

  // 更新成交量图表数据
  volumeChartOption.value.series = [
    {
      name: '成交量',
      type: 'line',
      data: reversedData.map((item) => item.volume),
      emphasis: { focus: 'series' },
      itemStyle: { color: '#1890ff' },
      symbol: 'none',
    },
  ]

  // 更新成交额图表数据
  turnoverChartOption.value.series = [
    {
      name: '成交额',
      type: 'line',
      data: reversedData.map((item) => parseFloat(item.turnover_amount)),
      emphasis: { focus: 'series' },
      itemStyle: { color: '#1890ff' },
      symbol: 'none',
    },
  ]

  // 更新振幅图表数据
  amplitudeChartOption.value.series = [
    {
      name: '振幅',
      type: 'line',
      data: reversedData.map((item) => parseFloat(item.amplitude)),
      emphasis: { focus: 'series' },
      itemStyle: { color: '#1890ff' },
      symbol: 'none',
    },
  ]

  // 更新涨跌幅图表数据
  changeChartOption.value.series = [
    {
      name: '涨跌幅',
      type: 'line',
      data: reversedData.map((item) => parseFloat(item.change_rate)),
      emphasis: { focus: 'series' },
      itemStyle: { color: '#1890ff' },
      symbol: 'none',
    },
  ]
}

// 搜索
const handleSearch = async () => {
  loading.value = true
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
  } finally {
    loading.value = false
  }
}

// 导出CSV
const handleExportCsv = async () => {
  loading.value = true
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
  } finally {
    loading.value = false
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
    tableData.value = [...tableData.value].sort((a, b) => {
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
