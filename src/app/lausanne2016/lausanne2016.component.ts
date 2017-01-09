import { Component, OnInit } from '@angular/core';
import { CsvReaderService } from '../csv-reader.service';

@Component({
  selector: 'app-lausanne2016',
  templateUrl: './lausanne2016.component.html',
  styleUrls: ['./lausanne2016.component.css']
})
export class Lausanne2016Component implements OnInit {
  public isDetailedStatisticalAnalysisCollapsed:boolean = true;
  public is10kmTukeyHSDTableCollapsed:boolean = true;
  public is21kmTukeyHSDTableCollapsed:boolean = true;
  public is42kmTukeyHSDTableCollapsed:boolean = true;

  constructor(private csvReader: CsvReaderService) {}

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

  ngOnInit() {
    this.csvReader.readCsvData('./assets/csv/marathon-lausanne-2016-speed-by-age.csv')
      .subscribe(data => {
        this.availableLabels.age = {
          label: 'Ages of runners',
          data: this.csvReader.getColumn(data, 'age')
        };
        this.availableSeries.speedDistance = {
          label: 'Average speed (m/s) by distance',
          data: [
            {data: this.csvReader.getColumn(data, '42 km'), label: '42 km - Speed (m/s)'},
            {data: this.csvReader.getColumn(data, '21 km'), label: '21 km - Speed (m/s)'},
            {data: this.csvReader.getColumn(data, '10 km'), label: '10 km - Speed (m/s)'}
          ]
        };
        this.availableSeries.speedSex = {
          label: 'Average speed (m/s) by sex',
          data: [
            {data: this.csvReader.getColumn(data, 'female'), label: 'Female runners - Speed (m/s)'},
            {data: this.csvReader.getColumn(data, 'male'), label: 'Male runners - Speed (m/s)'}
          ]
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
