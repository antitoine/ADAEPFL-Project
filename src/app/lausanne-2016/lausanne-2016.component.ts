import { Component, OnInit } from '@angular/core';
import { CsvReaderService } from '../util/csv-reader.service';

@Component({
  selector: 'app-lausanne-2016',
  templateUrl: './lausanne-2016.component.html',
  styleUrls: ['./lausanne-2016.component.css']
})
export class Lausanne2016Component implements OnInit {

  distributionByRunningType:string = 'time-distribution-42';

  isDetailedStatisticalAnalysisCollapsed:boolean = true;

  availableLabels: any = {};
  availableSeries: any = {};

  labelSelected: string;
  seriesSelected: string;

  chartLabels: number[] = null;
  chartSeries: Array<any> = null;
  chartLegend: boolean = true;
  chartType:string = 'line';
  chartOptions:any = {
    animationEasing: 'easeOutBounce',
    responsive: true
  };

  constructor(private csvReader: CsvReaderService) {}

  ngOnInit() {
    // Generate data for 'age' in x axis
    this.csvReader.readCsvData('./assets/csv/marathon-lausanne-2016-by-age.csv')
      .subscribe(data => {

        this.appendNewLabel('age', 'Ages of runners', this.csvReader.getColumn(data, 'age'));

        this.appendNewSeries('speed', 'age', [
          {data: this.csvReader.getColumn(data, 'speed'), label: 'Speed (m/s)'}
        ], 'Average speed (m/s)');

        this.appendNewSeries('speedDistance', 'age', [
          {data: this.csvReader.getColumn(data, '42km speed'), label: '42 km - Speed (m/s)'},
          {data: this.csvReader.getColumn(data, '21km speed'), label: '21 km - Speed (m/s)'},
          {data: this.csvReader.getColumn(data, '10km speed'), label: '10 km - Speed (m/s)'}
        ], 'Average speed (m/s) by distance');

        this.appendNewSeries('speedSex', 'age', [
          {data: this.csvReader.getColumn(data, 'female speed'), label: 'Female runners - Speed (m/s)'},
          {data: this.csvReader.getColumn(data, 'male speed'), label: 'Male runners - Speed (m/s)'}
        ], 'Average speed (m/s) by sex');

        this.appendNewSeries('count', 'age', [
          {data: this.csvReader.getColumn(data, 'count'), label: 'Number of runners'}
        ], 'Number of runners');

        this.appendNewSeries('countDistance', 'age', [
          {data: this.csvReader.getColumn(data, '42km count'), label: '42 km - Number of runners'},
          {data: this.csvReader.getColumn(data, '21km count'), label: '21 km - Number of runners'},
          {data: this.csvReader.getColumn(data, '10km count'), label: '10 km - Number of runners'}
        ], 'Number of runners by distance');

        this.appendNewSeries('countSex', 'age', [
          {data: this.csvReader.getColumn(data, 'female count'), label: 'Female runners - Number of runners'},
          {data: this.csvReader.getColumn(data, 'male count'), label: 'Male runners - Number of runners'}
        ], 'Number of runners by sex');

        this.appendNewSeries('time', 'age', [
          {data: this.csvReader.getColumn(data, 'time'), label: 'Speed (m/s)'}
        ], 'Average time (seconds)');

        this.appendNewSeries('timeDistance', 'age', [
          {data: this.csvReader.getColumn(data, '42km time'), label: '42 km - Time (s)'},
          {data: this.csvReader.getColumn(data, '21km time'), label: '21 km - Time (s)'},
          {data: this.csvReader.getColumn(data, '10km time'), label: '10 km - Time (s)'}
        ], 'Average time (seconds) by distance');

        this.appendNewSeries('timeSex', 'age', [
          {data: this.csvReader.getColumn(data, 'female time'), label: 'Female runners - Time (s)'},
          {data: this.csvReader.getColumn(data, 'male time'), label: 'Male runners - Time (s)'}
        ], 'Average time (seconds) by sex');

        this.labelSelected = 'age';
        this.onSelectLabelChange('age');
        this.seriesSelected = 'speed';
        this.onSelectSeriesChange('speed');
      });

    // Generate data for 'age' in x axis
    this.csvReader.readCsvData('./assets/csv/marathon-lausanne-2016-by-speed.csv')
      .subscribe(data => {
        this.appendNewLabel('speed', 'Speed (m/s) of runners', this.csvReader.getColumn(data, 'Speed (m/s) Rounded'));

        this.appendNewSeries('count', 'speed', [
          {data: this.csvReader.getColumn(data, 'count'), label: 'Number of runners'}
        ], 'Number of runners');

        this.appendNewSeries('countDistance', 'speed', [
          {data: this.csvReader.getColumn(data, '42km count'), label: '42 km - Number of runners'},
          {data: this.csvReader.getColumn(data, '21km count'), label: '21 km - Number of runners'},
          {data: this.csvReader.getColumn(data, '10km count'), label: '10 km - Number of runners'}
        ], 'Number of runners by distance');

        this.appendNewSeries('countSex', 'speed', [
          {data: this.csvReader.getColumn(data, 'female count'), label: 'Female runners - Number of runners'},
          {data: this.csvReader.getColumn(data, 'male count'), label: 'Male runners - Number of runners'}
        ], 'Number of runners by sex');

        this.appendNewSeries('time', 'speed', [
          {data: this.csvReader.getColumn(data, 'time'), label: 'Time (s)'}
        ], 'Average time (seconds)');

        this.appendNewSeries('timeDistance', 'speed', [
          {data: this.csvReader.getColumn(data, '42km time'), label: '42 km - Time (s)'},
          {data: this.csvReader.getColumn(data, '21km time'), label: '21 km - Time (s)'},
          {data: this.csvReader.getColumn(data, '10km time'), label: '10 km - Time (s)'}
        ], 'Average time (seconds) by distance');

        this.appendNewSeries('timeSex', 'speed', [
          {data: this.csvReader.getColumn(data, 'female time'), label: 'Female runners - Time (s)'},
          {data: this.csvReader.getColumn(data, 'male time'), label: 'Male runners - Time (s)'}
        ], 'Average time (seconds) by sex');
      });
  }

  onSelectLabelChange(key: string) {
    this.chartLabels = this.availableLabels[key].data;
    this.chartSeries = null;
    this.seriesSelected = null;
  }

  onSelectSeriesChange(key: string) {
    if (this.labelSelected) {
      this.chartSeries = this.availableSeries[key].series[this.labelSelected];
    }
  }

  /**
   * Use this function to append a new available label
   * @param key to access this label
   * @param name the displayed name in select
   * @param data
   */
  private appendNewLabel(key: string, name: string, data: any[]) {
    this.availableLabels[key] = {
      name: name,
      data: data
    };
  }

  /**
   * Use this function to append a new available Series for a given label
   * @param key to access this series
   * @param label to apply for
   * @param data
   * @param name set the displayed name in select
   */
  private appendNewSeries(key: string, label: string, data: any[], name: any = null) {
    this.availableSeries[key] = this.availableSeries[key] || {name: name, series: {}};
    this.availableSeries[key].series[label] = data;
  }
}
