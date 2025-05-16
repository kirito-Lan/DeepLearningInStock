<template>
  <div class="predict-container">
    <a-spin :spinning="loading || historyLoading || trainingLoading || exportingModel || exportingFeatures" tip="处理中...">
      <div class="card-container">
        <a-card :bordered="false" class="selection-card">
          <a-row :gutter="[16, 16]" >
            <a-col :xs="24" :sm="12" :md="12" :lg="10">
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
            <a-col :xs="24" :sm="12" :md="12" :lg="14">
              <a-form-item label="时间范围" class="mb-0">
                <a-range-picker
                  v-model:value="dateRange"
                  format="YYYY-MM-DD"
                  :placeholder="['开始日期', '结束日期']"
                  style="width: 100%"
                  :default-value="[dayjs().subtract(20, 'year'), dayjs()]"
                  @change="handleDateChange"
                >
                  <template #suffixIcon>
                    <calendar-outlined />
                  </template>
                </a-range-picker>
              </a-form-item>
            </a-col>
          </a-row>
          <a-divider style="margin-top: 16px; margin-bottom: 16px;" />
          <a-row :gutter="[8, 8]" justify="start">
            <a-col :xs="12" :sm="8" :md="6" :lg="4">
              <a-button type="primary" @click="showTrainModelModal" block :loading="trainingLoading">
                <template #icon><build-outlined /></template>
                训练模型
              </a-button>
            </a-col>
            <a-col :xs="12" :sm="8" :md="6" :lg="4">
              <a-button @click="fetchHistoricalData" block :loading="historyLoading || trainingLoading">
                <template #icon><history-outlined /></template>
                查询历史
              </a-button>
            </a-col>
            <a-col :xs="12" :sm="8" :md="6" :lg="4">
              <a-button @click="handleExportFeatures" block :loading="exportingFeatures">
                <template #icon><file-text-outlined /></template>
                导出特征
              </a-button>
            </a-col>
            <a-col :xs="12" :sm="8" :md="6" :lg="4">
              <a-button @click="handleExportModel" block :loading="exportingModel">
                <template #icon><file-zip-outlined /></template>
                导出模型
              </a-button>
            </a-col>
          </a-row>
        </a-card>

        <a-empty v-if="!historyLoading && !trainingLoading && !historicalPredictedData && !historicalMetrics" description="暂无历史数据，请选择参数后训练模型或查询历史" />

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

    <!-- Train Model Modal -->
    <a-modal
      v-model:open="isTrainModalVisible"
      title="配置模型训练参数"
      :confirm-loading="trainingLoading"
      centered
      @ok="handleModalTrainOk"
      @cancel="isTrainModalVisible = false"
      ok-text="开始训练"
      cancel-text="取消"
    >
      <a-form layout="vertical" :model="modalTrainFormState" ref="modalTrainFormRef">
        <a-form-item
          label="训练轮次 (Epoches)"
          name="epoches"
          :rules="[{ required: true, message: '请输入训练轮次' }]"
        >
          <a-input-number v-model:value="modalTrainFormState.epoches" :min="1" :max="2000" style="width: 100%" placeholder="例如: 150" />
        </a-form-item>
        <a-form-item
          label="正则化 (Reg)"
          name="reg"
          :rules="[{ required: true, message: '请输入正则化参数' }]"
        >
          <a-input-number v-model:value="modalTrainFormState.reg" :min="0.00001" :max="1.0" :step="0.0001" style="width: 100%" placeholder="例如: 0.001" />
        </a-form-item>
        <a-form-item
          label="Dropout"
          name="dropout"
          :rules="[{ required: true, message: '请输入Dropout率' }]"
        >
          <a-input-number v-model:value="modalTrainFormState.dropout" :min="0.0" :max="0.99" :step="0.01" style="width: 100%" placeholder="例如: 0.3" />
        </a-form-item>
      </a-form>
    </a-modal>

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
  BuildOutlined,
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
const dateRange = ref<[Dayjs, Dayjs]>([dayjs().subtract(20, 'year'), dayjs()]);
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

// Modal state and form
const isTrainModalVisible = ref(false);
interface ModalTrainFormState {
  epoches: number;
  reg: number;
  dropout: number;
}
const modalTrainFormRef = ref(); // For form validation
const modalTrainFormState = ref<ModalTrainFormState>({ // Reactive state for modal form
  epoches: 150,
  reg: 0.001,
  dropout: 0.3,
});

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

// Show Train Model Modal
const showTrainModelModal = () => {
  // Reset modal form to defaults each time it's opened for a fresh start
  modalTrainFormState.value = {
    epoches: 150,
    reg: 0.001,
    dropout: 0.3,
  };
  isTrainModalVisible.value = true;
};

// Handle Modal OK for Training
const handleModalTrainOk = async () => {
  if (!selectedStock.value) {
    message.warning('请先选择股票。');
    return;
  }
  if (!dateRange.value || dateRange.value.length < 2) {
    message.warning('请选择完整的时间范围。');
    return;
  }

  try {
    await modalTrainFormRef.value.validate(); // Validate modal form
  } catch (error) {
    message.warning('请填写所有必填的训练参数。');
    return; // Stop if validation fails
  }

  trainingLoading.value = true;
  try {
    const params = {
      stock_code: selectedStock.value,
      start_date: dateRange.value[0].format('YYYY-MM-DD'),
      end_date: dateRange.value[1].format('YYYY-MM-DD'),
      Epoches: modalTrainFormState.value.epoches,
      reg: modalTrainFormState.value.reg,
      dropout: modalTrainFormState.value.dropout, // Added dropout
    };
    const response = await trainModelPredictTrainModelPost(
      params as API.GetPredictRequest,
      { timeout: 1200000 } // 20 minutes in milliseconds
    );
    if (response.data && (response.data as any).code === 200 && (response.data as any).data === true) {
      isTrainModalVisible.value = false; // Close modal on success
      message.success('模型训练成功！将在3秒后获取最新的训练历史...');

      // Wait for 3 seconds before fetching historical data
      await new Promise(resolve => setTimeout(resolve, 3000));

      await fetchHistoricalData(); // Automatically fetch history after successful training and delay
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

// Confirm and Train Model (used by fetchHistoricalData if no data)
const confirmAndTrainModel = () => {
  Modal.confirm({
    title: '无历史数据',
    content: '未找到该股票在指定时间范围内的历史训练数据。是否现在配置参数并开始训练模型？',
    okText: '配置参数并训练',
    cancelText: '取消',
    centered: true,
    onOk: () => { // Changed from async and direct call
      showTrainModelModal(); // Open the modal for user to configure and then train
    },
    onCancel: () => {
      message.info('已取消模型训练。');
    },
  });
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
