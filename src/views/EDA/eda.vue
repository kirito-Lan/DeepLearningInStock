<template>
  <div class="eda-container">
    <a-spin
      :spinning="loading || descLoading || statsLoading || anomalyLoading || scatterLoading || correlationLoading"
      tip="数据加载中..."
    >
      <div class="card-container">
        <a-card :bordered="false" class="selection-card">
          <a-row :gutter="24">
            <a-col :span="10">
              <a-form-item label="股票选择" class="mb-0">
                <a-select
                  v-model:value="selectedStock"
                  :options="stockOptions"
                  placeholder="请选择股票"
                  show-search
                  :filter-option="filterOption"
                  style="width: 100%"
                  @change="handleStockChange"
                >
                  <template #suffixIcon>
                    <stock-outlined />
                  </template>
                </a-select>
              </a-form-item>
            </a-col>
            <a-col :span="12">
              <a-form-item label="时间范围" class="mb-0">
                <a-range-picker
                  v-model:value="dateRange"
                  format="YYYY-MM-DD"
                  :placeholder="['开始日期', '结束日期']"
                  style="width: 100%"
                  @change="handleDateChange"
                  :default-value="[dayjs().subtract(10, 'year'), dayjs()]"
                />
              </a-form-item>
            </a-col>
            <a-col :span="2">
              <a-button type="primary" :loading="loading" @click="fetchData">
                <template #icon><search-outlined /></template>
                查询
              </a-button>
            </a-col>
          </a-row>
        </a-card>
        <!-- 默认显示的面板 -->
        <a-collapse :default-active-key="['2', '3', '4', '5']">
          <a-collapse-panel key="1" header="数据描述">
            <a-card :bordered="false" class="data-description-card" :loading="descLoading">
              <template #extra>
                <a-tooltip title="数据集包含多种金融指标，本表显示每个指标的基本解释">
                  <info-circle-outlined style="color: #1890ff" />
                </a-tooltip>
              </template>
              <a-table
                :columns="columns"
                :data-source="dataDescription"
                :pagination="false"
                size="middle"
                :scroll="{ y: 400 }"
                :row-key="(record) => record.key"
              >
                <template #bodyCell="{ column, text }">
                  <template v-if="column.dataIndex === 'indicatorIcon'">
                    <component :is="getIconByKey(text)" style="font-size: 18px" />
                  </template>
                </template>
              </a-table>
            </a-card>
          </a-collapse-panel>
          <!-- 股票的统计性描述 -->
          <a-collapse-panel key="2" header="股票的统计性描述">
            <a-card :bordered="false" class="statistics-card" :loading="statsLoading">
              <a-table
                :columns="statsColumns"
                :data-source="stockStatistics"
                :pagination="false"
                size="middle"
                :scroll="{ y: 400 }"
                :row-key="(record) => record.key"
              >
                <template #bodyCell="{ column, text }">
                  <template v-if="column.dataIndex === 'value'">
                    {{ text }}
                  </template>
                </template>
              </a-table>
            </a-card>
          </a-collapse-panel>

          <a-collapse-panel key="3" header="交易量异常值检测">
            <a-card :bordered="false" class="anomaly-section-inner-card" :loading="anomalyLoading">
              <v-chart :option="anomalyChartOption" autoresize class="chart-container" />
            </a-card>
          </a-collapse-panel>

          <!-- 散点图面板 -->
          <a-collapse-panel key="4" header="宏观经济指标与股价散点图分析">
            <a-card
              :bordered="false"
              class="scatter-plot-card"
              :loading="scatterLoading"
            >
              <a-row :gutter="16">
                <a-col :span="8">
                  <v-chart :option="cpiScatterOption" autoresize class="scatter-chart-item" />
                </a-col>
                <a-col :span="8">
                  <v-chart :option="ppiScatterOption" autoresize class="scatter-chart-item" />
                </a-col>
                <a-col :span="8">
                  <v-chart :option="pmiScatterOption" autoresize class="scatter-chart-item" />
                </a-col>
              </a-row>
            </a-card>
          </a-collapse-panel>

          <!-- 时序图面板 -->
          <a-collapse-panel key="5" header="宏观经济指标与股价时序图">
            <a-card
              :bordered="false"
              class="correlation-line-chart-card"
              :loading="correlationLoading"
            >
              <a-row :gutter="16">
                <a-col :span="24">
                  <v-chart :option="cpiLineChartOption" autoresize class="line-chart-item" />
                </a-col>
                <a-col :span="24">
                  <v-chart :option="ppiLineChartOption" autoresize class="line-chart-item" />
                </a-col>
                <a-col :span="24">
                  <v-chart :option="pmiLineChartOption" autoresize class="line-chart-item" />
                </a-col>
              </a-row>
            </a-card>
          </a-collapse-panel>
        </a-collapse>

        <a-empty
          v-if="
            !loading &&
            !descLoading &&
            !statsLoading &&
            !anomalyLoading &&
            !scatterLoading &&
            !correlationLoading &&
            dataDescription.length === 0 &&
            stockStatistics.length === 0 &&
            (!anomalyChartOption.series ||
              !(anomalyChartOption.series as any)[0] ||
              !(anomalyChartOption.series as any)[0].data ||
              (anomalyChartOption.series as any)[0].data.length === 0) &&
            (!cpiScatterOption.series ||
              !(cpiScatterOption.series as any)[0] ||
              !(cpiScatterOption.series as any)[0].data ||
              (cpiScatterOption.series as any)[0].data.length === 0) &&
            (!cpiLineChartOption.series ||
              !(cpiLineChartOption.series as any)[0] ||
              !(cpiLineChartOption.series as any)[0].data ||
              (cpiLineChartOption.series as any)[0].data.length === 0)
          "
          description="暂无数据，请选择股票和时间范围进行查询"
        />
      </div>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { message } from 'ant-design-vue'
import type { Dayjs } from 'dayjs'
import dayjs from 'dayjs'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, ScatterChart, BarChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  MarkPointComponent,
  DataZoomComponent,
  VisualMapComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import type { EChartsOption, SeriesOption } from 'echarts'

// Ant Design Icons
import {
  StockOutlined,
  SearchOutlined,
  InfoCircleOutlined,
  LineChartOutlined as IconLineChartOutlined,
  AreaChartOutlined,
  BarChartOutlined as IconBarChartOutlined,
  PieChartOutlined,
  FundOutlined,
  DotChartOutlined,
  FieldTimeOutlined,
} from '@ant-design/icons-vue'

// API Functions
import {
  descriptDataSetPredictDescriptionGet,
  stockAnalysisPredictStockAnalysisPost,
  anomalyDetectionPredictAnomalyDetectionPost,
  scatterPlotPredictScatterPlotPost,
  correlationAnalysisPredictCorrelationAnalysisPost,
} from '@/api/predict' // Assuming this path is correct for the project
import { getStockListStockGetStockListGet } from '@/api/stock'

use([
  CanvasRenderer,
  LineChart,
  ScatterChart,
  BarChart,
  TitleComponent,
  LegendComponent,
  GridComponent,
  VisualMapComponent,
  TooltipComponent,
  MarkPointComponent,
  DataZoomComponent,
])

// 组件名称定义
defineOptions({
  name: 'StockEDAView',
})

// 定义类型
interface StockOption {
  value: string
  label: string
}

interface DataDescriptionItem {
  key: string
  indicatorName: string
  indicatorNameCn: string
  indicatorDescription: string
  indicatorIcon: string
}

interface StatisticsRow {
  key: string
  name: string
  sample_num?: number | string
  min_value?: number | string
  max_value?: number | string
  avg?: number | string
  std?: number | string
  var?: number | string
  bias?: number | string
  sharp?: number | string
}

interface AnomalyData {
  volume: number
  trade_date: string
  Volume_outlier: boolean
}

interface AnomalySeriesDataPoint {
  value: number
  trade_date: string
  isOutlier: boolean
}

interface ScatterPoint {
  CPI: number
  PPI: number
  PMI: number
  Close: number
}

interface CorrelationDataItem {
  Date: string
  CPI: number | null
  PPI: number | null
  PMI: number | null
  Close: number | null
}

// 数据 Refs
const loading = ref(false)
const descLoading = ref(false)
const statsLoading = ref(false)
const anomalyLoading = ref(false)
const scatterLoading = ref(false)
const correlationLoading = ref(false)

const selectedStock = ref<string>('')
const dateRange = ref<[Dayjs, Dayjs]>([dayjs().subtract(20, 'year'), dayjs()])
const stockOptions = ref<StockOption[]>([])
const stockStatistics = ref<StatisticsRow[]>([])
const dataDescription = ref<DataDescriptionItem[]>([])
const correlationData = ref<CorrelationDataItem[]>([])

const columns = [
  { title: '', dataIndex: 'indicatorIcon', key: 'indicatorIcon', width: 80 },
  { title: '指标名称', dataIndex: 'indicatorNameCn', key: 'indicatorNameCn', width: 120 },
  { title: '英文名称', dataIndex: 'indicatorName', key: 'indicatorName', width: 120 },
  { title: '指标描述', dataIndex: 'indicatorDescription', key: 'indicatorDescription' },
]

const statsColumns = [
  {
    title: '统计类别',
    dataIndex: 'name',
    key: 'name',
    width: 100,
    align: 'center',
    className: 'stats-column',
  },
  {
    title: '样本数',
    dataIndex: 'sample_num',
    key: 'sample_num',
    align: 'center',
    className: 'stats-column',
  },
  {
    title: '最小值',
    dataIndex: 'min_value',
    key: 'min_value',
    align: 'center',
    className: 'stats-column',
  },
  {
    title: '最大值',
    dataIndex: 'max_value',
    key: 'max_value',
    align: 'center',
    className: 'stats-column',
  },
  { title: '均值', dataIndex: 'avg', key: 'avg', align: 'center', className: 'stats-column' },
  { title: '标准差', dataIndex: 'std', key: 'std', align: 'center', className: 'stats-column' },
  { title: '方差', dataIndex: 'var', key: 'var', align: 'center', className: 'stats-column' },
  { title: '偏度', dataIndex: 'bias', key: 'bias', align: 'center', className: 'stats-column' },
  { title: '峰度', dataIndex: 'sharp', key: 'sharp', align: 'center', className: 'stats-column' },
]

const initialAnomalyChartOption: EChartsOption = {
  title: { text: '', left: 'center' },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#1890ff',
    borderWidth: 1,
    textStyle: { color: '#333', fontSize: 13 },
    padding: [10, 15],
    extraCssText: 'box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);',
    formatter: (params: any) => {
      if (!params || params.length === 0) return ''
      const pointInfo = params[0]
      const seriesData = pointInfo.data as AnomalySeriesDataPoint
      const date = seriesData.trade_date
      const volume = seriesData.value
      const isOutlier = seriesData.isOutlier
      let html = `<div style="font-weight: bold; margin-bottom: 8px; padding-bottom: 5px; border-bottom: 1px solid #eee;">📅 ${date}</div>`
      html += `<div style="display: flex; align-items: center; margin-bottom: 4px;"><span style="display:inline-block; margin-right:8px; border-radius:50%; width:10px; height:10px; background-color:${pointInfo.color};"></span>交易量：<strong>${volume.toLocaleString()}</strong></div>`
      if (isOutlier) {
        html += `<div style="color: #ff4d4f; font-weight: bold; margin-top: 5px;">⚠️ 检测到异常</div>`
      } else {
        html += `<div style="color: #52c41a; margin-top: 5px;">✔️ 状态正常</div>`
      }
      return html
    },
  },
  legend: { data: ['交易量'], bottom: 10 },
  grid: { left: '3%', right: '4%', bottom: '15%', top: '10%', containLabel: true },
  xAxis: {
    type: 'category',
    data: [] as string[],
    name: '交易日期',
    boundaryGap: false,
    axisLabel: { rotate: 45, interval: 'auto' as number | 'auto' },
  },
  yAxis: { type: 'value', name: '交易量', scale: true },
  series: [
    {
      name: '交易量',
      type: 'line',
      data: [] as AnomalySeriesDataPoint[],
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2, color: '#1890ff' },
      markPoint: {
        symbol: 'circle',
        symbolSize: 8,
        label: { show: false },
        itemStyle: { color: '#ff4d4f' },
        data: [] as any[],
      },
    } as SeriesOption,
  ],
  dataZoom: [
    { type: 'slider', show: true, xAxisIndex: [0], start: 0, end: 100 },
    { type: 'inside', xAxisIndex: [0], start: 0, end: 100 },
  ],
}
const anomalyChartOption = ref<EChartsOption>(JSON.parse(JSON.stringify(initialAnomalyChartOption)))

const baseScatterOption = (titleText: string, xAxisName: string): EChartsOption => ({
  title: { text: titleText, left: 'center', top: 10, textStyle: { fontSize: 16 } },
  grid: { left: '10%', right: '10%', bottom: '15%', top: '20%', containLabel: false },
  tooltip: {
    trigger: 'item',
    formatter: (params: any) => {
      if (!params.value || params.value.length < 2) return '数据不完整'
      return `${xAxisName}: ${params.value[0].toFixed(2)}<br/>收盘价: ${params.value[1].toFixed(2)}`
    },
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#1890ff',
    borderWidth: 1,
    textStyle: { color: '#333', fontSize: 13 },
    padding: [10, 15],
  },
  xAxis: {
    type: 'value',
    name: xAxisName,
    nameLocation: 'middle',
    nameGap: 25,
    scale: true,
    splitLine: { show: false },
    axisLabel: { formatter: (v: number) => v.toFixed(2) },
  },
  yAxis: {
    type: 'value',
    name: '收盘价',
    scale: true,
    splitLine: { show: true, lineStyle: { type: 'dashed' } },
    axisLabel: { formatter: (v: number) => v.toFixed(2) },
  },
  series: [
    {
      name: `${titleText}`,
      type: 'scatter',
      symbolSize: 8,
      data: [] as [number, number][],
      itemStyle: { color: '#1890ff' },
    } as SeriesOption,
  ],
  dataZoom: [
    { type: 'slider', show: true, xAxisIndex: 0, yAxisIndex: 0, bottom: 10, height: 20 },
    { type: 'inside', xAxisIndex: 0, yAxisIndex: 0 },
  ],
})

const cpiScatterOption = ref<EChartsOption>(baseScatterOption('收盘价 vs CPI', 'CPI 值'))
const ppiScatterOption = ref<EChartsOption>(baseScatterOption('收盘价 vs PPI', 'PPI 值'))
const pmiScatterOption = ref<EChartsOption>(baseScatterOption('收盘价 vs PMI', 'PMI 值'))

// Base option for new line charts
const baseLineChartOption = (
  titleText: string,
  metricKey: 'CPI' | 'PPI' | 'PMI',
  metricDisplayName: string,
): EChartsOption => ({
  title: { text: titleText, left: 'center', top: 10, textStyle: { fontSize: 16, fontWeight: 'normal' } },
  tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'cross', crossStyle: { color: '#999' } },
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#1890ff',
    borderWidth: 1,
    textStyle: { color: '#333', fontSize: 13 },
    padding: [10, 15],
    formatter: (params: any) => {
      if (!params || params.length === 0) return ''
      let html = `<div style="font-weight: bold; margin-bottom: 8px; padding-bottom: 5px; border-bottom: 1px solid #eee;">📅 ${params[0].axisValue}</div>`
      params.forEach((param: any) => {
        const value = param.value !== null && param.value !== undefined ? param.value.toFixed(2) : '-'
        html += `<div style="display: flex; align-items: center; margin-bottom: 4px;"><span style="display:inline-block; margin-right:8px; border-radius:50%; width:10px; height:10px; background-color:${param.color};"></span>${param.seriesName}: <strong>${value}</strong></div>`
      })
      return html
    },
  },
  legend: { data: [metricDisplayName, '收盘价'], bottom: 10, textStyle: { fontSize: 12 } },
  grid: { left: '5%', right: '5%', bottom: '15%', top: '18%', containLabel: true },
  xAxis: [
    {
      type: 'category',
      data: [],
      axisPointer: { type: 'shadow' },
      axisLabel: { rotate: 30, interval: 'auto' as number | 'auto', fontSize: 11 },
    },
  ],
  yAxis: [
    {
      type: 'value',
      name: metricDisplayName,
      min: (value) => (value.min * 0.95).toFixed(2),
      max: (value) => (value.max * 1.05).toFixed(2),
      axisLabel: { formatter: '{value}', fontSize: 11 },
      nameTextStyle: { fontSize: 12, padding: [0, 0, 0, 30] },
      splitLine: { lineStyle: { type: 'dashed', color: '#eee'} },
    },
    {
      type: 'value',
      name: '收盘价',
      min: (value) => (value.min * 0.9).toFixed(0), // Adjusted for potentially larger scale
      max: (value) => (value.max * 1.1).toFixed(0), // Adjusted for potentially larger scale
      axisLabel: { formatter: '{value}', fontSize: 11 },
      nameTextStyle: { fontSize: 12, padding: [0, 30, 0, 0] },
      splitLine: { show: false }, // Avoid clutter with the other Y-axis grid
    },
  ],
  series: [
    {
      name: metricDisplayName,
      type: 'line',
      yAxisIndex: 0,
      data: [],
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2 },
      itemStyle: { color: '#5470C6' }
    },
    {
      name: '收盘价',
      type: 'line',
      yAxisIndex: 1,
      data: [],
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2 },
      itemStyle: { color: '#91CC75'}
    },
  ],
  dataZoom: [
    { type: 'slider', show: true, xAxisIndex: [0], start: 0, end: 100, bottom: 30, height: 20 },
    { type: 'inside', xAxisIndex: [0], start: 0, end: 100 },
  ],
})

const cpiLineChartOption = ref<EChartsOption>(baseLineChartOption('CPI 与收盘价 时序图', 'CPI', 'CPI'))
const ppiLineChartOption = ref<EChartsOption>(baseLineChartOption('PPI 与收盘价 时序图', 'PPI', 'PPI'))
const pmiLineChartOption = ref<EChartsOption>(baseLineChartOption('PMI 与收盘价 时序图', 'PMI', 'PMI'))

// 工具函数
const getIconByKey = (key: string) => {
  const iconMap: Record<string, any> = {
    Close: IconLineChartOutlined,
    Volume: IconBarChartOutlined,
    Open: FundOutlined,
    High: AreaChartOutlined,
    Low: DotChartOutlined,
    CPI: PieChartOutlined,
    PPI: PieChartOutlined,
    PMI: FieldTimeOutlined,
    default: IconLineChartOutlined,
  }
  return iconMap[key] || iconMap.default
}

const filterOption = (input: string, option: unknown) => {
  const opt = option as { label: string }
  return (opt?.label ?? '').toLowerCase().includes(input.toLowerCase())
}

// 数据获取与更新函数
const fetchStockList = async () => {
  try {
    const response = await getStockListStockGetStockListGet()
    const result = (response.data as any)?.data
    if (result && Array.isArray(result)) {
      stockOptions.value = result.map((stock: { name: string; code: string }) => ({
        value: stock.code,
        label: stock.name,
      }))
    } else {
      message.error('获取股票列表失败或数据格式不正确')
    }
  } catch (error) {
    console.error('获取股票列表错误:', error)
    message.error('获取股票列表错误')
  }
}

const fetchDataDescription = async () => {
  descLoading.value = true
  try {
    const response = await descriptDataSetPredictDescriptionGet()
    const result = (response.data as any)?.data
    if (result && typeof result === 'object' && !Array.isArray(result)) {
      const formattedData: DataDescriptionItem[] = []
      Object.entries(result).forEach(([key, valueObj]) => {
        const typedValue = valueObj as { 中文名称: string; 指数解释: string }
        formattedData.push({
          key,
          indicatorName: key,
          indicatorNameCn: typedValue['中文名称'] || '-',
          indicatorDescription: typedValue['指数解释'] || '-',
          indicatorIcon: key,
        })
      })
      dataDescription.value = formattedData
    } else {
      message.error('获取数据描述失败或数据格式不正确')
      dataDescription.value = []
    }
  } catch (error) {
    console.error('获取数据描述错误:', error)
    message.error('获取数据描述错误')
    dataDescription.value = []
  } finally {
    descLoading.value = false
  }
}

const fetchStockStatistics = async () => {
  if (!selectedStock.value) return
  statsLoading.value = true
  try {
    const response = await stockAnalysisPredictStockAnalysisPost({
      stock_code: selectedStock.value,
      start_date: dateRange.value[0].format('YYYY-MM-DD'),
      end_date: dateRange.value[1].format('YYYY-MM-DD'),
    })
    const result = (response.data as any)?.data
    if (result && typeof result === 'object' && !Array.isArray(result)) {
      stockStatistics.value = [
        {
          key: 'row1',
          name: '基本统计',
          sample_num: result.sample_num,
          min_value: result.min_value,
          max_value: result.max_value,
          avg: result.avg,
        },
        {
          key: 'row2',
          name: '高阶统计',
          std: result.std,
          var: result.var,
          bias: result.bias,
          sharp: result.sharp,
        },
      ]
    } else {
      message.error('获取股票统计数据失败或格式不正确')
      stockStatistics.value = []
    }
  } catch (error) {
    console.error('获取股票统计数据错误:', error)
    message.error('获取股票统计数据错误')
    stockStatistics.value = []
  } finally {
    statsLoading.value = false
  }
}

const updateAnomalyChart = (data: AnomalyData[]) => {
  const newOption = JSON.parse(JSON.stringify(anomalyChartOption.value)) as EChartsOption
  if (!data || data.length === 0) {
    if (newOption.xAxis) (newOption.xAxis as any).data = []
    if (newOption.series && (newOption.series as any)[0]) {
      ;(newOption.series as any)[0].data = []
      if ((newOption.series as any)[0].markPoint) {
        ;(newOption.series as any)[0].markPoint.data = []
      }
    }
    anomalyChartOption.value = newOption
    return
  }

  const dates = data.map((item) => item.trade_date)
  let intervalSetting: number | 'auto' = 'auto'
  if (dates.length > 60) intervalSetting = Math.floor(dates.length / 20)
  else if (dates.length > 30) intervalSetting = Math.floor(dates.length / 10)
  else if (dates.length > 15) intervalSetting = Math.floor(dates.length / 5)
  else if (dates.length > 0) intervalSetting = 0

  const seriesData: AnomalySeriesDataPoint[] = data.map((item) => ({
    value: item.volume,
    trade_date: item.trade_date,
    isOutlier: item.Volume_outlier,
  }))

  const markPointsData = data
    .filter((item) => item.Volume_outlier)
    .map((item) => ({
      name: '异常点',
      xAxis: item.trade_date,
      yAxis: item.volume,
    }))

  if (newOption.xAxis) {
    ;(newOption.xAxis as any).data = dates
    if ((newOption.xAxis as any).axisLabel)
      (newOption.xAxis as any).axisLabel.interval = intervalSetting
  }
  if (newOption.series && (newOption.series as any)[0]) {
    ;(newOption.series as any)[0].data = seriesData
    if ((newOption.series as any)[0].markPoint) {
      ;(newOption.series as any)[0].markPoint.data = markPointsData
    }
  }
  anomalyChartOption.value = newOption
}

const fetchAnomalyData = async () => {
  if (!selectedStock.value) {
    updateAnomalyChart([])
    return
  }
  anomalyLoading.value = true
  try {
    const response = await anomalyDetectionPredictAnomalyDetectionPost({
      stock_code: selectedStock.value,
      start_date: dateRange.value[0].format('YYYY-MM-DD'),
      end_date: dateRange.value[1].format('YYYY-MM-DD'),
    })
    const chartDataPayload = (response.data as any)?.data as AnomalyData[] | undefined
    if (chartDataPayload && Array.isArray(chartDataPayload)) {
      updateAnomalyChart(chartDataPayload)
    } else {
      message.error('获取异常值检测数据失败或无数据')
      updateAnomalyChart([])
    }
  } catch (error) {
    console.error('获取异常值检测数据错误:', error)
    message.error('获取异常值检测数据错误')
    updateAnomalyChart([])
  } finally {
    anomalyLoading.value = false
  }
}

const updateScatterCharts = (data: ScatterPoint[] | null) => {
  const updateSingleScatter = (
    optionRef: import('vue').Ref<EChartsOption>,
    key: keyof ScatterPoint,
  ) => {
    if (optionRef.value.series && (optionRef.value.series as any)[0]) {
      ;(optionRef.value.series as any)[0].data = data ? data.map((p) => [p[key], p.Close]) : []
    } else {
      optionRef.value.series = [
        { data: data ? data.map((p) => [p[key], p.Close]) : [] } as SeriesOption,
      ]
    }
  }
  if (!data || data.length === 0) {
    if (cpiScatterOption.value.series && (cpiScatterOption.value.series as any)[0])
      (cpiScatterOption.value.series as any)[0].data = []
    if (ppiScatterOption.value.series && (ppiScatterOption.value.series as any)[0])
      (ppiScatterOption.value.series as any)[0].data = []
    if (pmiScatterOption.value.series && (pmiScatterOption.value.series as any)[0])
      (pmiScatterOption.value.series as any)[0].data = []
    return
  }
  updateSingleScatter(cpiScatterOption, 'CPI')
  updateSingleScatter(ppiScatterOption, 'PPI')
  updateSingleScatter(pmiScatterOption, 'PMI')
  scatterLoading.value = false
}

const fetchScatterData = async () => {
  if (!selectedStock.value) {
    updateScatterCharts(null)
    return
  }
  scatterLoading.value = true
  try {
    const response = await scatterPlotPredictScatterPlotPost({
      stock_code: selectedStock.value,
      start_date: dateRange.value[0].format('YYYY-MM-DD'),
      end_date: dateRange.value[1].format('YYYY-MM-DD'),
    })
    const scatterDataPayload = (response.data as any)?.data as ScatterPoint[] | undefined
    if (scatterDataPayload && Array.isArray(scatterDataPayload)) {
      updateScatterCharts(scatterDataPayload)
    } else {
      message.error('获取散点图数据失败或无数据')
      updateScatterCharts(null)
    }
  } catch (error) {
    console.error('获取散点图数据错误:', error)
    message.error('获取散点图数据错误')
    updateScatterCharts(null)
  } finally {
    scatterLoading.value = false
  }
}

const updateCorrelationCharts = (data: CorrelationDataItem[] | null) => {
  const dates = data ? data.map((item) => item.Date) : []

  const updateSingleLineChart = (
    optionRef: import('vue').Ref<EChartsOption>,
    metricKey: 'CPI' | 'PPI' | 'PMI',
  ) => {
    const metricValues = data ? data.map((item) => item[metricKey]) : []
    const closeValues = data ? data.map((item) => item.Close) : []

    if (optionRef.value.xAxis && Array.isArray(optionRef.value.xAxis)) {
      ;(optionRef.value.xAxis[0] as any).data = dates
      let intervalSetting: number | 'auto' = 'auto';
      if (dates.length > 120) intervalSetting = Math.floor(dates.length / 15);
      else if (dates.length > 60) intervalSetting = Math.floor(dates.length / 10);
      else if (dates.length > 30) intervalSetting = Math.floor(dates.length / 5);
      else if (dates.length > 0) intervalSetting = 0;
      (optionRef.value.xAxis[0] as any).axisLabel.interval = intervalSetting;

    }
    if (optionRef.value.series && Array.isArray(optionRef.value.series)) {
      ;(optionRef.value.series[0] as any).data = metricValues
      ;(optionRef.value.series[1] as any).data = closeValues
    }
     // Force chart to re-render with new options
    optionRef.value = { ...optionRef.value };
  }
  if (!data || data.length === 0) {
      const emptyDates: string[] = [];
      const emptyValues: null[] = [];
      if (cpiLineChartOption.value.xAxis && Array.isArray(cpiLineChartOption.value.xAxis)) (cpiLineChartOption.value.xAxis[0] as any).data = emptyDates;
      if (cpiLineChartOption.value.series && Array.isArray(cpiLineChartOption.value.series)) {
          (cpiLineChartOption.value.series[0] as any).data = emptyValues;
          (cpiLineChartOption.value.series[1] as any).data = emptyValues;
      }
      cpiLineChartOption.value = { ...cpiLineChartOption.value };

      if (ppiLineChartOption.value.xAxis && Array.isArray(ppiLineChartOption.value.xAxis)) (ppiLineChartOption.value.xAxis[0] as any).data = emptyDates;
      if (ppiLineChartOption.value.series && Array.isArray(ppiLineChartOption.value.series)) {
          (ppiLineChartOption.value.series[0] as any).data = emptyValues;
          (ppiLineChartOption.value.series[1] as any).data = emptyValues;
      }
       ppiLineChartOption.value = { ...ppiLineChartOption.value };

      if (pmiLineChartOption.value.xAxis && Array.isArray(pmiLineChartOption.value.xAxis)) (pmiLineChartOption.value.xAxis[0] as any).data = emptyDates;
      if (pmiLineChartOption.value.series && Array.isArray(pmiLineChartOption.value.series)) {
          (pmiLineChartOption.value.series[0] as any).data = emptyValues;
          (pmiLineChartOption.value.series[1] as any).data = emptyValues;
      }
      pmiLineChartOption.value = { ...pmiLineChartOption.value };
      return;
  }

  updateSingleLineChart(cpiLineChartOption, 'CPI')
  updateSingleLineChart(ppiLineChartOption, 'PPI')
  updateSingleLineChart(pmiLineChartOption, 'PMI')
}

const fetchCorrelationData = async () => {
  if (!selectedStock.value) {
    updateCorrelationCharts(null)
    return
  }
  correlationLoading.value = true
  try {
    const response = await correlationAnalysisPredictCorrelationAnalysisPost({
      stock_code: selectedStock.value,
      start_date: dateRange.value[0].format('YYYY-MM-DD'),
      end_date: dateRange.value[1].format('YYYY-MM-DD'),
    })
    const correlationResult = (response.data as any)?.data as CorrelationDataItem[] | undefined
    if (correlationResult && Array.isArray(correlationResult)) {
      correlationData.value = correlationResult
      updateCorrelationCharts(correlationResult)
    } else {
      message.error('获取时序图数据失败或无数据')
      updateCorrelationCharts(null)
    }
  } catch (error) {
    console.error('获取时序图数据错误:', error)
    message.error('获取时序图数据错误')
    updateCorrelationCharts(null)
  } finally {
    correlationLoading.value = false
  }
}

const fetchData = async () => {
  if (!selectedStock.value) {
    message.warning('请选择股票')
    return
  }
  if (!dateRange.value || dateRange.value.length < 2) {
    message.warning('请选择完整的时间范围')
    return
  }
  loading.value = true
  dataDescription.value = []
  stockStatistics.value = []
  updateAnomalyChart([])
  updateScatterCharts(null)
  updateCorrelationCharts(null)
  await Promise.allSettled([
    fetchDataDescription(),
    fetchStockStatistics(),
    fetchAnomalyData(),
    fetchScatterData(),
    fetchCorrelationData(),
  ])
  loading.value = false
}

const handleStockChange = (value: string) => {
  selectedStock.value = value
  if (value) {
    fetchData()
  } else {
    dataDescription.value = []
    stockStatistics.value = []
    updateAnomalyChart([])
    updateScatterCharts(null)
    updateCorrelationCharts(null)
  }
}

const handleDateChange = () => {
  if (selectedStock.value && dateRange.value && dateRange.value.length === 2) {
    fetchData()
  }
}

onMounted(async () => {
  await fetchStockList()

  // 如果股票列表不为空，并且当前没有选中的股票，则自动选择第一个
  if (stockOptions.value.length > 0 && !selectedStock.value) {
    selectedStock.value = stockOptions.value[0].value
  }

  // 如果有选中的股票 (无论是预选的还是刚自动选的) 且日期有效
  if (selectedStock.value && dateRange.value && dateRange.value.length === 2) {
    await fetchData() // fetchData 会处理所有相关的 loading 状态
  } else {
    // 如果条件不满足（例如股票列表为空或日期范围无效），则清空数据和图表
    dataDescription.value = []
    stockStatistics.value = []
    updateAnomalyChart([])
    updateScatterCharts(null)
    updateCorrelationCharts(null)
    if (stockOptions.value.length === 0) {
      message.info('股票列表为空，无法自动加载数据。')
    } else if (!selectedStock.value) {
      // 此情况理论上不应在 stockOptions.value.length > 0 时发生，作为后备
      message.warning('未能确定要查询的股票。')
    }
  }
})

onUnmounted(() => {
  /* Cleanup if necessary */
})
</script>

<style scoped>
.eda-container {
  padding: 24px;
  background-color: #f0f2f5;
  min-height: 100vh;
}
.card-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.selection-card,
.data-description-card,
.statistics-card,
.anomaly-section-inner-card,
.scatter-plot-card,
.correlation-line-chart-card {
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.09);
}
.mb-0 {
  margin-bottom: 0;
}
:deep(.stats-column) {
  font-size: 14px;
  font-weight: 500;
  color: #333;
}
:deep(.ant-table-thead > tr > th) {
  background-color: #fafafa;
  font-weight: 600;
  color: #333;
}
:deep(.ant-table-tbody > tr > td) {
  padding: 12px 8px;
}
:deep(.ant-table-tbody > tr:hover > td) {
  background-color: #e6f7ff;
}
.chart-container,
.scatter-chart-item,
.line-chart-item {
  width: 100%;
  height: 400px;
  min-height: 400px;
}
:deep(.ant-collapse-content-box) {
  padding: 0 !important;
}
:deep(.ant-card-body) {
  padding: 20px !important;
  overflow: visible !important;
}
.anomaly-section-inner-card .ant-card-body {
  padding-top: 12px !important;
}
.scatter-chart-item {
  height: 350px;
  min-height: 350px;
}
.line-chart-item {
  width: 100%;
  height: 380px;
  min-height: 380px;
}
</style>
