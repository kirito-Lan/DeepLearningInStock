<template>
  <div class="predict-container">
    <a-spin :spinning="loading || historyLoading || trainingLoading || exportingModel || exportingFeatures" tip="处理中...">
      <div class="card-container">
        <a-card :bordered="false" class="selection-card">
          <a-row :gutter="[16, 16]" >
            <a-col :xs="24" :sm="12" :md="12" :lg="6">
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
            <a-col :xs="24" :sm="12" :md="12" :lg="8">
              <a-form-item label="时间范围" class="mb-0">
                <a-range-picker
                  v-model:value="dateRange"
                  format="YYYY-MM-DD"
                  :placeholder="['开始日期', '结束日期']"
                  style="width: 100%"
                  :default-value="[dayjs().subtract(3, 'year'), dayjs()]"
                  @change="handleDateChange"
                >
                  <template #suffixIcon>
                    <calendar-outlined />
                  </template>
                </a-range-picker>
              </a-form-item>
            </a-col>
            <a-col :xs="24" :sm="12" :md="12" :lg="5">
              <a-form-item class="mb-0">
                <template #label>
                  <dashboard-outlined /> 训练轮次
                </template>
                <a-input-number
                  v-model:value="epoches"
                  :min="1"
                  :max="1000"
                  style="width: 100%"
                  placeholder="例如: 150"
                />
              </a-form-item>
            </a-col>
            <a-col :xs="24" :sm="12" :md="12" :lg="5">
              <a-form-item class="mb-0">
                <template #label>
                  <control-outlined /> 正则化
                </template>
                <a-input-number
                  v-model:value="reg"
                  :min="0.0001"
                  :max="1.0"
                  :step="0.0001"
                  style="width: 100%"
                  placeholder="例如: 0.001"
                  :formatter="(value: number | string) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')"
                  :parser="(value: string) => parseFloat(value.replace(/\$\s?|(,*)/g, ''))"
                />
              </a-form-item>
            </a-col>
          </a-row>
          <a-divider style="margin-top: 16px; margin-bottom: 16px;" />
          <a-row :gutter="[8, 8]" justify="start"> 
            <a-col :xs="12" :sm="8" :md="6" :lg="3">
              <a-button type="primary" :loading="loading" @click="fetchPredictionData" block>
                <template #icon><search-outlined /></template>
                执行预测
              </a-button>
            </a-col>
            <a-col :xs="12" :sm="8" :md="6" :lg="3">
              <a-button :loading="historyLoading || trainingLoading" @click="fetchHistoricalData" block>
                <template #icon><history-outlined /></template>
                查询历史
              </a-button>
            </a-col>
            <a-col :xs="12" :sm="8" :md="6" :lg="3">
              <a-button :loading="exportingModel" @click="handleExportModel" block>
                <template #icon><file-zip-outlined /></template>
                导出模型
              </a-button>
            </a-col>
            <a-col :xs="12" :sm="8" :md="6" :lg="3">
              <a-button :loading="exportingFeatures" @click="handleExportFeatures" block>
                <template #icon><file-text-outlined /></template>
                导出特征
              </a-button>
            </a-col>
          </a-row>
        </a-card>

        <!-- Prediction results (original placeholder) -->
        <a-empty v-if="!loading && !predictionResult && !historicalPredictedData && !historicalMetrics" description="暂无数据，请选择参数后点击按钮操作" />
        <a-card v-if="predictionResult && !historyLoading" class="prediction-result-card">
           <template #title>
            <fund-projection-screen-outlined /> 预测结果 (主)
          </template>
          <!-- Placeholder for chart or data display -->
          <p>{{ predictionResult }}</p>
        </a-card>

        <!-- Historical Data Panels -->
        <a-collapse v-model:activeKey="activeHistoryCollapseKeys" class="history-collapse">
          <a-collapse-panel key="historyMetrics" class="history-panel">
            <template #header>
              <table-outlined /> 训练历史 - 评估指标
            </template>
            <a-card :bordered="false" :loading="historyLoading">
              <a-table
                v-if="historicalMetrics && Object.keys(historicalMetrics).length > 0"
                :columns="metricsTableColumns"
                :data-source="metricsTableDataSource"
                :pagination="false"
                bordered
                size="small"
                row-key="metric"
              />
              <a-empty v-else description="无评估指标数据，请先获取历史训练数据或训练模型" />
            </a-card>
          </a-collapse-panel>

          <a-collapse-panel key="historyLossChart" class="history-panel">
            <template #header>
              <line-chart-outlined /> 训练过程 - 损失曲线
            </template>
            <a-card :bordered="false" :loading="historyLoading">
              <v-chart v-if="historicalLossData && historicalLossData.length > 0" :option="lossChartOption" autoresize class="history-chart-item" />
              <a-empty v-else description="无损失曲线数据" />
            </a-card>
          </a-collapse-panel>

          <a-collapse-panel key="historyChart" class="history-panel">
            <template #header>
              <line-chart-outlined /> 训练历史 - 实际 vs. 预测
            </template>
            <a-card :bordered="false" :loading="historyLoading">
              <v-chart v-if="historicalPredictedData && historicalPredictedData.length > 0" :option="historyChartOption" autoresize class="history-chart-item" />
              <a-empty v-else description="无图表数据，请先获取历史训练数据或训练模型" />
            </a-card>
          </a-collapse-panel>
        </a-collapse>

      </div>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue';
import { message, Modal } from 'ant-design-vue';
import type { Dayjs } from 'dayjs';
import dayjs from 'dayjs';
import { use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { LineChart } from 'echarts/charts';
import {
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  DataZoomComponent,
} from 'echarts/components';
import VChart from 'vue-echarts';
import type { EChartsOption, SeriesOption } from 'echarts';
import type { AxiosResponse } from 'axios';

// Ant Design Icons
import {
  StockOutlined,
  SearchOutlined,
  CalendarOutlined,
  DashboardOutlined,
  ControlOutlined,
  FundProjectionScreenOutlined,
  HistoryOutlined,
  LineChartOutlined,
  TableOutlined,
  FileZipOutlined,
  FileTextOutlined,
} from '@ant-design/icons-vue';

// API Functions
import { getStockListStockGetStockListGet } from '@/api/stock';
import {
  getTrainedScorePredictGetTrainedScorePost,
  trainModelPredictTrainModelPost,
  exportModelPredictExportModelGet,
  getFeaturedFilePredictGetFeaturedFilePost,
} from '@/api/predict';
// Assuming predict API function will be imported here later
// import { yourPredictApiFunction } from '@/api/predict';

use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
  LegendComponent,
  GridComponent,
  TooltipComponent,
  DataZoomComponent,
]);

// Component name definition
defineOptions({
  name: 'StockPredictView',
});

// Define types
interface StockOption {
  value: string;
  label: string;
}

// Data Refs
const loading = ref(false);
const selectedStock = ref<string | undefined>(undefined);
const stockOptions = ref<StockOption[]>([]);
const dateRange = ref<[Dayjs, Dayjs]>([dayjs().subtract(3, 'year'), dayjs()]);
const epoches = ref<number>(150);
const reg = ref<number>(0.001);

const predictionResult = ref<any>(null); // Placeholder for prediction result

// New state variables for historical data
const historyLoading = ref(false);
const trainingLoading = ref(false);
const exportingModel = ref(false);
const exportingFeatures = ref(false);
const historicalPredictedData = ref<Array<{ Date: string; Actual: number; Predicted: number }>>([]);
const historicalMetrics = ref<Record<string, number> | null>(null);
const historicalLossData = ref<Array<{ Epochs: number; Train_Loss: number; Validate_Loss: number }>>([]);
const activeHistoryCollapseKeys = ref<string[]>(['historyMetrics', 'historyLossChart', 'historyChart']);

const historyChartOption = ref<EChartsOption>({});
const lossChartOption = ref<EChartsOption>({});
const metricsTableColumns = ref<Array<{ title: string; dataIndex: string; key: string; align?: string }>>([]);
const metricsTableDataSource = ref<Array<Record<string, number | string>>>([]);

// Filter option for select
const filterOption = (input: string, option: unknown) => {
  const opt = option as { label: string };
  return (opt?.label ?? '').toLowerCase().includes(input.toLowerCase());
};

// Fetch stock list
const fetchStockList = async () => {
  loading.value = true;
  try {
    const response = await getStockListStockGetStockListGet();
    const result = (response.data as any)?.data;
    if (result && Array.isArray(result)) {
      stockOptions.value = result.map((stock: { name: string; code: string }) => ({
        value: stock.code,
        label: `${stock.name} (${stock.code})`,
      }));
      if (stockOptions.value.length > 0) {
        selectedStock.value = stockOptions.value[0].value; // Default to first stock
      }
    } else {
      message.error('获取股票列表失败或数据格式不正确');
      stockOptions.value = [];
    }
  } catch (error) {
    console.error('获取股票列表错误:', error);
    message.error('获取股票列表错误');
    stockOptions.value = [];
  } finally {
    loading.value = false;
  }
};

const handleStockChange = (value: string) => {
  selectedStock.value = value;
  predictionResult.value = null; // Clear previous main prediction results
  historicalPredictedData.value = []; // Clear historical data
  historicalMetrics.value = null;
  historicalLossData.value = []; // Clear loss data
  activeHistoryCollapseKeys.value = ['historyMetrics', 'historyLossChart', 'historyChart'];
};

const handleDateChange = () => {
  predictionResult.value = null;
  historicalPredictedData.value = [];
  historicalMetrics.value = null;
  historicalLossData.value = []; // Clear loss data
  activeHistoryCollapseKeys.value = ['historyMetrics', 'historyLossChart', 'historyChart'];
};

// Update history chart
const updateHistoryChart = () => {
  if (!historicalPredictedData.value || historicalPredictedData.value.length === 0) {
    historyChartOption.value = { series: [] }; // Clear chart if no data
    return;
  }
  const dates = historicalPredictedData.value.map(item => item.Date);
  const actuals = historicalPredictedData.value.map(item => item.Actual);
  const predicts = historicalPredictedData.value.map(item => item.Predicted);

  let intervalSetting: number | 'auto' = 'auto';
  const maxLabels = 15; // Max desired labels on x-axis
  if (dates.length > maxLabels * 2.5) { // Heuristic for dense data
      intervalSetting = Math.floor(dates.length / maxLabels);
  } else if (dates.length > 0 && dates.length <= maxLabels) {
      intervalSetting = 0; // Show all labels if few data points
  }

  historyChartOption.value = {
    title: { text: '实际值 vs. 预测值', left: 'center', textStyle: { fontWeight: 'normal' } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: { data: ['实际值', '预测值'], bottom: 10 },
    grid: { left: '5%', right: '4%', bottom: '15%', containLabel: true }, // Adjusted bottom for dataZoom
    xAxis: { type: 'category', boundaryGap: false, data: dates, axisLabel: { rotate: 30, interval: intervalSetting } },
    yAxis: { type: 'value', scale: true, axisLabel: { formatter: '{value}' } },
    series: [
      {
        name: '实际值',
        type: 'line',
        data: actuals,
        smooth: true,
        showSymbol: false,
        itemStyle: { color: '#5470C6' },
      },
      {
        name: '预测值',
        type: 'line',
        data: predicts,
        smooth: true,
        showSymbol: false,
        itemStyle: { color: '#91CC75' },
      },
    ],
    dataZoom: [
      { type: 'slider', show: true, xAxisIndex: [0], start: 0, end: 100, bottom: 30 }, // Explicit bottom
      { type: 'inside', xAxisIndex: [0], start: 0, end: 100 },
    ],
  };
};

// Update metrics table
const updateMetricsTable = () => {
  if (!historicalMetrics.value) {
    metricsTableDataSource.value = [];
    metricsTableColumns.value = [];
    return;
  }

  const metricOrder = ['mse', 'rmse', 'mae', 'r2'];
  const metricDisplayNames: Record<string, string> = {
    mse: '均方误差 (MSE)',
    rmse: '均方根误差 (RMSE)',
    mae: '平均绝对误差 (MAE)',
    r2: 'R² 分数',
  };

  const newColumns: Array<{ title: string; dataIndex: string; key: string; align?: string }> = [];
  const singleRowData: Record<string, number | string> = { key: 'metricsRow' };

  metricOrder.forEach(key => {
    if (historicalMetrics.value && historicalMetrics.value[key] !== undefined) {
      newColumns.push({
        title: metricDisplayNames[key] || key.toUpperCase(),
        dataIndex: key,
        key: key,
        align: 'center' as 'center',
      });
      singleRowData[key] = typeof historicalMetrics.value[key] === 'number' 
        ? parseFloat((historicalMetrics.value[key] as number).toFixed(4)) 
        : historicalMetrics.value[key];
    }
  });
  
  metricsTableColumns.value = newColumns;
  metricsTableDataSource.value = newColumns.length > 0 ? [singleRowData] : [];
};

// Update Loss Chart
const updateLossChart = () => {
  if (!historicalLossData.value || historicalLossData.value.length === 0) {
    lossChartOption.value = { series: [] }; // Clear chart if no data
    return;
  }
  const epochs = historicalLossData.value.map(item => item.Epochs);
  const trainLoss = historicalLossData.value.map(item => item.Train_Loss);
  const validateLoss = historicalLossData.value.map(item => item.Validate_Loss);

  lossChartOption.value = {
    title: { text: '训练/验证损失曲线', left: 'center', textStyle: { fontWeight: 'normal' } },
    tooltip: { 
      trigger: 'axis', 
      axisPointer: { type: 'cross', label: { backgroundColor: '#6a7985' } },
      formatter: (params: any) => {
        if (!params || params.length === 0 || params[0].axisValue == null) return '';
        let html = `Epoch: ${params[0].axisValueLabel || params[0].axisValue}<br/>`;
        params.forEach((param: any) => {
          const value = Array.isArray(param.value) && param.value.length > 1 ? param.value[1] : param.value;
          html += `${param.marker} ${param.seriesName}: ${value !== undefined && value !== null ? parseFloat(value).toFixed(6) : '-'}<br/>`;
        });
        return html;
      }
    },
    legend: { data: ['训练损失', '验证损失'], bottom: 10 },
    grid: { left: '5%', right: '4%', bottom: '15%', containLabel: true },
    xAxis: { 
      type: 'value', 
      name: 'Epochs',
      boundaryGap: false, 
      min: 1,
      axisLabel: { interval: 'auto' }
    },
    yAxis: { type: 'value', name: 'Loss', scale: true, axisLabel: { formatter: (v: number) => v.toFixed(4) } },
    series: [
      {
        name: '训练损失',
        type: 'line',
        data: epochs.map((epoch, index) => [epoch, trainLoss[index]]),
        smooth: true,
        showSymbol: false,
        sampling: 'lttb',
        itemStyle: { color: '#c23531' }, 
        zlevel: 1
      },
      {
        name: '验证损失',
        type: 'line',
        data: epochs.map((epoch, index) => [epoch, validateLoss[index]]),
        smooth: true,
        showSymbol: false,
        sampling: 'lttb',
        itemStyle: { color: '#2f4554' },
        zlevel: 1
      },
    ],
    dataZoom: [
      { type: 'slider', show: true, xAxisIndex: [0], start: 0, end: 100, bottom: 30 },
      { type: 'inside', xAxisIndex: [0], start: 0, end: 100 },
    ],
    progressive: 200,
    progressiveThreshold: 1000,
    hoverLayerThreshold: 3000,
  };
};

// Helper function for file download
const handleFileDownload = (response: AxiosResponse, defaultFilename: string) => {
  const contentType = response.headers['content-type'] || 'application/octet-stream';
  let filename = defaultFilename;
  const disposition = response.headers['content-disposition'];
  if (disposition && disposition.indexOf('attachment') !== -1) {
    const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
    const matches = filenameRegex.exec(disposition);
    if (matches != null && matches[1]) {
      filename = matches[1].replace(/['"]/g, '');
    }
  }

  const blob = new Blob([response.data], { type: contentType });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
  message.success(`已开始下载 ${filename}`);
};

// Export Model
const handleExportModel = async () => {
  if (!selectedStock.value) {
    message.warning('请先选择股票以导出对应的模型。');
    return;
  }
  exportingModel.value = true;
  try {
    const response = await exportModelPredictExportModelGet({ code: selectedStock.value });
    // Check for actual response data and headers for file download
    if (response.data && response.headers && response.headers['content-type'] === 'application/octet-stream') {
       handleFileDownload(response as AxiosResponse, `model_${selectedStock.value}.keras`);
    } else if ((response.data as any)?.code === 40400) {
      message.error((response.data as any)?.msg || '模型文件不存在，请先训练模型。');
    } else {
      // Handle cases where it might be a JSON response with an error message but not 40400
      if ((response.data as any)?.msg) {
         message.error(`导出模型失败: ${(response.data as any).msg}`);
      } else {
         message.error('导出模型失败，响应不是有效的文件流。');
      }
      console.error('Export model failed, unexpected response:', response);
    }
  } catch (error: any) {
    if (error.response && error.response.data && error.response.data.code === 40400) {
      message.error(error.response.data.msg || '模型文件不存在，请先训练模型。');
    } else if (error.response && error.response.data && error.response.data.msg) {
      message.error(`导出模型错误: ${error.response.data.msg}`);
    } else {
      message.error('导出模型请求失败，请检查网络或联系管理员。');
    }
    console.error('导出模型错误:', error);
  } finally {
    exportingModel.value = false;
  }
};

// Export Features
const handleExportFeatures = async () => {
  if (!selectedStock.value) {
    message.warning('请选择股票。');
    return;
  }
  if (!dateRange.value || dateRange.value.length < 2) {
    message.warning('请选择完整的时间范围以导出对应的特征数据。');
    return;
  }
  exportingFeatures.value = true;
  try {
    const params = {
      stock_code: selectedStock.value,
      start_date: dateRange.value[0].format('YYYY-MM-DD'),
      end_date: dateRange.value[1].format('YYYY-MM-DD'),
    };
    const response = await getFeaturedFilePredictGetFeaturedFilePost(params as API.GetPredictRequest);
    if (response.data && response.headers && response.headers['content-type'] === 'application/octet-stream') {
      handleFileDownload(response as AxiosResponse, `features_${selectedStock.value}_${params.start_date}_to_${params.end_date}.csv`);
    } else if ((response.data as any)?.code === 40400) {
      message.error((response.data as any)?.msg || '特征工程文件不存在。请先确保在选定日期范围内有数据和特征或已完成模型训练。');
    } else {
       if ((response.data as any)?.msg) {
         message.error(`导出特征数据失败: ${(response.data as any).msg}`);
      } else {
        message.error('导出特征数据失败，响应不是有效的文件流。');
      }
      console.error('Export features failed, unexpected response:', response);
    }
  } catch (error: any) {
     if (error.response && error.response.data && error.response.data.code === 40400) {
      message.error(error.response.data.msg || '特征工程文件不存在。');
    } else if (error.response && error.response.data && error.response.data.msg) {
      message.error(`导出特征数据错误: ${error.response.data.msg}`);
    } else {
      message.error('导出特征数据请求失败，请检查网络或联系管理员。');
    }
    console.error('导出特征数据错误:', error);
  } finally {
    exportingFeatures.value = false;
  }
};

// Train Model
const handleTrainModel = async () => {
  if (!selectedStock.value || !dateRange.value || dateRange.value.length < 2 || !epoches.value || !reg.value) {
    message.error('请确保已选择股票、时间范围并已输入训练轮次和正则化参数。');
    return;
  }
  trainingLoading.value = true;
  try {
    const params = {
      stock_code: selectedStock.value,
      start_date: dateRange.value[0].format('YYYY-MM-DD'),
      end_date: dateRange.value[1].format('YYYY-MM-DD'),
      Epoches: epoches.value,
      reg: reg.value,
    };
    const response = await trainModelPredictTrainModelPost(params as API.GetPredictRequest);
    if (response.data && (response.data as any).code === 200 && (response.data as any).data === true) {
      message.success('模型训练成功！正在获取最新的训练历史...');
      await fetchHistoricalData(); // Automatically fetch history after successful training
    } else {
      message.error((response.data as any)?.msg || '模型训练失败，请检查参数或稍后再试。');
    }
  } catch (error: any) {
    console.error('模型训练请求错误:', error);
    message.error(error?.data?.msg || error?.message || '模型训练过程中发生错误。');
  } finally {
    trainingLoading.value = false;
  }
};

// Confirm and Train Model
const confirmAndTrainModel = () => {
  Modal.confirm({
    title: '无历史数据',
    content: '未找到该股票在指定时间范围内的历史训练数据。是否现在开始训练模型？',
    okText: '开始训练',
    cancelText: '取消',
    centered: true,
    onOk: async () => {
      await handleTrainModel();
    },
    onCancel: () => {
      message.info('已取消模型训练。');
    },
  });
};

// Fetch Historical Data
const fetchHistoricalData = async () => {
  if (!selectedStock.value) {
    message.warning('请选择股票');
    return;
  }
  if (!dateRange.value || dateRange.value.length < 2) {
    message.warning('请选择完整的时间范围');
    return;
  }

  historyLoading.value = true;
  historicalPredictedData.value = []; // Clear previous data
  historicalMetrics.value = null;
  historicalLossData.value = []; // Clear previous loss data
  try {
    const params = {
      stock_code: selectedStock.value,
      start_date: dateRange.value[0].format('YYYY-MM-DD'),
      end_date: dateRange.value[1].format('YYYY-MM-DD'),
      // Epoches and reg are not sent for getTrainedScore
    };
    const response = await getTrainedScorePredictGetTrainedScorePost(params as API.GetPredictRequest);

    if (response.data && (response.data as any).code === 200) {
      const resultData = (response.data as any).data;
      if (resultData && resultData.predicted_data && resultData.metrics && resultData.loss_data) {
        historicalPredictedData.value = resultData.predicted_data;
        historicalMetrics.value = resultData.metrics;
        historicalLossData.value = resultData.loss_data;
        updateHistoryChart();
        updateMetricsTable();
        updateLossChart();
        message.success('历史训练数据加载成功！');
        activeHistoryCollapseKeys.value = ['historyMetrics', 'historyLossChart', 'historyChart'];
      } else {
        message.warning('获取到历史数据，但格式不完整 (可能缺少损失数据)。');
        if(resultData?.predicted_data) historicalPredictedData.value = resultData.predicted_data;
        if(resultData?.metrics) historicalMetrics.value = resultData.metrics;
        if(resultData?.loss_data) historicalLossData.value = resultData.loss_data;
        updateHistoryChart();
        updateMetricsTable();
        updateLossChart();
        activeHistoryCollapseKeys.value = ['historyMetrics', 'historyLossChart', 'historyChart'];
      }
    } else if (response.data && (response.data as any).code === 40400) {
      message.info((response.data as any).msg || '没有历史预测数据。');
      confirmAndTrainModel();
      activeHistoryCollapseKeys.value = ['historyMetrics', 'historyLossChart', 'historyChart'];
    } else {
      message.error((response.data as any)?.msg || '获取历史训练数据失败。');
      activeHistoryCollapseKeys.value = ['historyMetrics', 'historyLossChart', 'historyChart'];
    }
  } catch (error: any) {
    console.error('获取历史训练数据错误:', error);
    message.error(error?.data?.msg || error?.message || '获取历史训练数据时发生错误。');
    activeHistoryCollapseKeys.value = ['historyMetrics', 'historyLossChart', 'historyChart'];
  } finally {
    historyLoading.value = false;
  }
};

// Fetch prediction data (original placeholder function)
const fetchPredictionData = async () => {
  if (!selectedStock.value) {
    message.warning('请选择股票');
    return;
  }
  if (!dateRange.value || dateRange.value.length < 2) {
    message.warning('请选择完整的时间范围');
    return;
  }
  if (epoches.value === null || epoches.value === undefined || epoches.value <= 0) {
    message.warning('请输入有效的训练轮次');
    return;
  }
   if (reg.value === null || reg.value === undefined || reg.value <= 0) {
    message.warning('请输入有效的正则化参数');
    return;
  }

  loading.value = true;
  predictionResult.value = null; // Clear previous results

  console.log('Fetching prediction with params:', {
    stock_code: selectedStock.value,
    start_date: dateRange.value[0].format('YYYY-MM-DD'),
    end_date: dateRange.value[1].format('YYYY-MM-DD'),
    Epoches: epoches.value,
    reg: reg.value,
  });

  // Simulate API call
  // Replace with actual API call:
  // try {
  //   const params = {
  //     stock_code: selectedStock.value,
  //     start_date: dateRange.value[0].format('YYYY-MM-DD'),
  //     end_date: dateRange.value[1].format('YYYY-MM-DD'),
  //     Epoches: epoches.value,
  //     reg: reg.value,
  //   };
  //   const response = await yourPredictApiFunction(params);
  //   predictionResult.value = response.data; // Adjust based on actual response structure
  //   message.success('预测成功');
  // } catch (error) {
  //   console.error('预测请求错误:', error);
  //   message.error('预测请求失败');
  //   predictionResult.value = null;
  // } finally {
  //   loading.value = false;
  // }

  // Placeholder for demonstration
  setTimeout(() => {
    predictionResult.value = {
      message: '这是模拟的预测结果',
      params: {
        stock_code: selectedStock.value,
        start_date: dateRange.value[0].format('YYYY-MM-DD'),
        end_date: dateRange.value[1].format('YYYY-MM-DD'),
        Epoches: epoches.value,
        reg: reg.value,
      }
    };
    loading.value = false;
    message.success('模拟预测完成');
  }, 1500);
};

onMounted(async () => {
  await fetchStockList();
  // Any other initial data fetching for the predict view if needed
});

</script>

<style scoped>
.predict-container {
  padding: 24px;
  background-color: #f0f2f5;
  min-height: 100vh;
}
.card-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.selection-card {
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.09);
}
.prediction-result-card {
  margin-top: 24px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.09);
}
.history-collapse {
  margin-top: 24px;
}
.history-panel .ant-card-body {
  padding: 0px !important; /* Adjust padding for cards within collapse */
}
.history-chart-item {
  width: 100%;
  height: 400px; 
  min-height: 350px;
}
.mb-0 {
  margin-bottom: 0;
}
:deep(.ant-form-item-label > label) {
  font-weight: 500;
  display: inline-flex;
  align-items: center;
}
:deep(.ant-form-item-label .anticon) {
  margin-right: 6px;
  font-size: 16px;
}

/* Center align labels in the first row of parameters */
.selection-card .ant-row:first-child .ant-form-item-label > label {
  width: 100%; /* Ensure the label takes full width to allow text-align to work */
  text-align: center;
  justify-content: center; /* For flex-aligned labels with icons */
}

/* Add more styles as needed, similar to eda.vue */
</style>
