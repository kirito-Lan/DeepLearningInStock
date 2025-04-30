 <template>
  <div class="stock-data-container">
    <a-card>
      <div class="search-container">
        <a-row :gutter="16" class="search-row" justify="center">
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
        <a-radio-group v-model:value="chartType" button-style="solid" class="chart-type-selector">
          <a-radio-button value="price">价格走势</a-radio-button>
          <a-radio-button value="volume">成交量</a-radio-button>
          <a-radio-button value="turnover">成交额</a-radio-button>
          <a-radio-button value="amplitude">振幅</a-radio-button>
          <a-radio-button value="change">涨跌幅</a-radio-button>
        </a-radio-group>
        <v-chart class="chart" :option="chartOption" autoresize />
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

    <a-modal v-model:visible="detailVisible" title="股票详情" :footer="null" width="800px">
      <a-descriptions :column="2" bordered>
        <a-descriptions-item label="交易日期">{{ detailData.trade_date }}</a-descriptions-item>
        <a-descriptions-item label="开盘价">{{ detailData.open_price }}</a-descriptions-item>
        <a-descriptions-item label="收盘价">{{ detailData.close_price }}</a-descriptions-item>
        <a-descriptions-item label="最高价">{{ detailData.high_price }}</a-descriptions-item>
        <a-descriptions-item label="最低价">{{ detailData.low_price }}</a-descriptions-item>
        <a-descriptions-item label="成交量">{{ detailData.volume }}</a-descriptions-item>
        <a-descriptions-item label="成交额">{{ detailData.turnover_amount }}</a-descriptions-item>
        <a-descriptions-item label="振幅">{{ detailData.amplitude }}%</a-descriptions-item>
        <a-descriptions-item label="涨跌幅">{{ detailData.change_rate }}%</a-descriptions-item>
        <a-descriptions-item label="涨跌额">{{ detailData.change_amount }}</a-descriptions-item>
        <a-descriptions-item label="换手率">{{ detailData.turnover_rate }}%</a-descriptions-item>
      </a-descriptions>
    </a-modal>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, watch } from 'vue'
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
import { getStockListStockGetStockListGet } from '@/api/stock'
import { getStockDataStockGetStockDataPost } from '@/api/stock'
import { getStockCsvStockGetStockCsvPost } from '@/api/stock'
import { batchExportToExcelCommonBatchExportToExcelPost } from '@/api/common'
import type { StockDataItem } from '@/typings/stock'

// 注册必要的组件
use([CanvasRenderer, LineChart, TitleComponent, TooltipComponent, LegendComponent, GridComponent])

// 搜索表单
const searchForm = ref({
  stock_code: '',
  start_date: '',
  end_date: '',
})

const dateRange = ref<[Dayjs, Dayjs]>([dayjs().subtract(1, 'year'), dayjs()])
const stockList = ref<{ name: string; code: string }[]>([])
const chartType = ref('price')

// 图表配置
const chartOption = ref({
  tooltip: {
    trigger: 'axis',
  },
  legend: {
    data: [] as string[],
  },
  xAxis: {
    type: 'category',
    data: [] as string[],
  },
  yAxis: {
    type: 'value',
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

// 更新图表数据
const updateChartData = () => {
  const dates = tableData.value.map((item) => item.trade_date)
  chartOption.value.xAxis.data = dates

  switch (chartType.value) {
    case 'price':
      chartOption.value.legend.data = ['开盘价', '收盘价', '最高价', '最低价']
      chartOption.value.series = [
        {
          name: '开盘价',
          type: 'line',
          data: tableData.value.map((item) => parseFloat(item.open_price)),
        },
        {
          name: '收盘价',
          type: 'line',
          data: tableData.value.map((item) => parseFloat(item.close_price)),
        },
        {
          name: '最高价',
          type: 'line',
          data: tableData.value.map((item) => parseFloat(item.high_price)),
        },
        {
          name: '最低价',
          type: 'line',
          data: tableData.value.map((item) => parseFloat(item.low_price)),
        },
      ]
      break
    case 'volume':
      chartOption.value.legend.data = ['成交量']
      chartOption.value.series = [
        {
          name: '成交量',
          type: 'line',
          data: tableData.value.map((item) => item.volume),
        },
      ]
      break
    case 'turnover':
      chartOption.value.legend.data = ['成交额']
      chartOption.value.series = [
        {
          name: '成交额',
          type: 'line',
          data: tableData.value.map((item) => parseFloat(item.turnover_amount)),
        },
      ]
      break
    case 'amplitude':
      chartOption.value.legend.data = ['振幅']
      chartOption.value.series = [
        {
          name: '振幅',
          type: 'line',
          data: tableData.value.map((item) => parseFloat(item.amplitude)),
        },
      ]
      break
    case 'change':
      chartOption.value.legend.data = ['涨跌幅']
      chartOption.value.series = [
        {
          name: '涨跌幅',
          type: 'line',
          data: tableData.value.map((item) => parseFloat(item.change_rate)),
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
const handleTableChange = (pag: { current: number; pageSize: number }, filters: any, sorter: any) => {
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

.chart-container {
  height: 400px;
  margin-bottom: 24px;
}

.chart {
  height: 100%;
  width: 100%;
}

.chart-type-selector {
  margin-bottom: 16px;
}
</style>