using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ColorGanInterface
{
	public partial class Form1 : Form
	{
		private static readonly HttpClient Client = new HttpClient();
		private const string Pix2PixUrl = "http://localhost:5000/pix2pix";
		
		public Form1()
		{
			InitializeComponent();
		}

		private async void button1_Click(object sender, EventArgs e)
		{
			var open = new OpenFileDialog();  
			open.Filter = @"Image Files(*.jpg; *.jpeg, *png)|*.jpg; *.jpeg; *.png;";  
			if (open.ShowDialog() == DialogResult.OK) {  
				pictureBoxGray.Image = new Bitmap(open.FileName);   
			}  
			
			var request = new HttpRequestMessage(HttpMethod.Post, Pix2PixUrl);
			var content = new MultipartFormDataContent();
			
			byte[] byteArray = null;
			using(var memoryStream = new MemoryStream())
			{
				pictureBoxGray.Image.Save(memoryStream, ImageFormat.Jpeg);
				byteArray = memoryStream.ToArray();
			}

			content.Add(new ByteArrayContent(byteArray), "file", "file.jpg");
			request.Content = content;

			var response = await Client.SendAsync(request);
			response.EnsureSuccessStatusCode();

			var responseByArray = await response.Content.ReadAsByteArrayAsync();
			
			using (var ms = new MemoryStream(responseByArray))
			{
				pictureBoxColor.Image = Image.FromStream(ms);
			}
		}
	}
}