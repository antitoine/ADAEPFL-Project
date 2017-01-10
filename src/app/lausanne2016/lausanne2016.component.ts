import { Component, OnInit } from '@angular/core';
import { CsvReaderService } from '../csv-reader.service';

@Component({
  selector: 'app-lausanne2016',
  templateUrl: './lausanne2016.component.html',
  styleUrls: ['./lausanne2016.component.css']
})
export class Lausanne2016Component implements OnInit {

  isDetailedStatisticalAnalysisCollapsed:boolean = true;

  availableLabels: any = {};
  availableSeries: any = {};

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
    this.csvReader.readCsvData('./assets/csv/marathon-lausanne-2016-by-age.csv')
      .subscribe(data => {
        this.availableLabels.age = {
          name: 'Ages of runners',
          data: this.csvReader.getColumn(data, 'age'),
          series: ['speed', 'speedDistance', 'speedSex', 'count', 'countDistance', 'countSex']
        };
        this.availableSeries.speed = {
          name: 'Average speed (m/s)',
          data: [
            {data: this.csvReader.getColumn(data, 'speed'), label: 'Speed (m/s)'}
          ],
          labels: ['age']
        };
        this.availableSeries.speedDistance = {
          name: 'Average speed (m/s) by distance',
          data: [
            {data: this.csvReader.getColumn(data, '42km speed'), label: '42 km - Speed (m/s)'},
            {data: this.csvReader.getColumn(data, '21km speed'), label: '21 km - Speed (m/s)'},
            {data: this.csvReader.getColumn(data, '10km speed'), label: '10 km - Speed (m/s)'}
          ],
          labels: ['age']
        };
        this.availableSeries.speedSex = {
          name: 'Average speed (m/s) by sex',
          data: [
            {data: this.csvReader.getColumn(data, 'female speed'), label: 'Female runners - Speed (m/s)'},
            {data: this.csvReader.getColumn(data, 'male speed'), label: 'Male runners - Speed (m/s)'}
          ],
          labels: ['age']
        };
        this.availableSeries.count = {
          name: 'Number of runners',
          data: [
            {data: this.csvReader.getColumn(data, 'count'), label: 'Number of runners'}
          ],
          labels: ['age']
        };
        this.availableSeries.countDistance = {
          name: 'Number of runners by distance',
          data: [
            {data: this.csvReader.getColumn(data, '42km count'), label: '42 km - Number of runners'},
            {data: this.csvReader.getColumn(data, '21km count'), label: '21 km - Number of runners'},
            {data: this.csvReader.getColumn(data, '10km count'), label: '10 km - Number of runners'}
          ],
          labels: ['age']
        };
        this.availableSeries.countSex = {
          name: 'Number of runners by sex',
          data: [
            {data: this.csvReader.getColumn(data, 'female count'), label: 'Female runners - Number of runners'},
            {data: this.csvReader.getColumn(data, 'male count'), label: 'Male runners - Number of runners'}
          ],
          labels: ['age']
        };
        this.availableSeries.time = {
          name: 'Average time (seconds)',
          data: [
            {data: this.csvReader.getColumn(data, 'time'), label: 'Speed (m/s)'}
          ],
          labels: ['age']
        };
        this.availableSeries.timeDistance = {
          name: 'Average time (seconds) by distance',
          data: [
            {data: this.csvReader.getColumn(data, '42km time'), label: '42 km - Time (s)'},
            {data: this.csvReader.getColumn(data, '21km time'), label: '21 km - Time (s)'},
            {data: this.csvReader.getColumn(data, '10km time'), label: '10 km - Time (s)'}
          ],
          labels: ['age']
        };
        this.availableSeries.timeSex = {
          name: 'Average time (seconds) by sex',
          data: [
            {data: this.csvReader.getColumn(data, 'female time'), label: 'Female runners - Time (s)'},
            {data: this.csvReader.getColumn(data, 'male time'), label: 'Male runners - Time (s)'}
          ],
          labels: ['age']
        };
      });
  }

  onSelectLabelChange(key: string) {
    this.chartLabels = this.availableLabels[key].data;
  }

  onSelectSeriesChange(key: string) {
    this.chartSeries = this.availableSeries[key].data;
  }
}
