/*
 * MIT License
 *
 * Copyright (c) 2024 Shubham Panchal
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package android.example.clip

import android.example.clip.ui.components.AppProgressDialog
import android.example.clip.ui.components.hideProgressDialog
import android.example.clip.ui.components.setProgressDialogText
import android.example.clip.ui.components.showProgressDialog
import android.example.clip.ui.theme.ClipcppTheme
import android.graphics.Bitmap
import android.graphics.BitmapFactory.*
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {

            val viewModel = viewModel<MainActivityViewModel>()

            ClipcppTheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    topBar = {
                        TopAppBar(
                            title = { Text(text = "clip.android") },
                            actions = {
                                Row {
                                    IconButton(onClick = {
                                        viewModel.showModelInfo()
                                    }) {
                                        Icon(
                                            imageVector = Icons.Default.Info,
                                            contentDescription = "Model Info"
                                        )
                                    }
                                }
                            }
                        )
                    }
                ) { innerPadding ->
                    Column(modifier = Modifier.padding(innerPadding)) {
                        SelectImagePanel(viewModel)
                        EnterDescriptionPanel(viewModel)
                    }
                    LoadModelProgressDialog(viewModel)
                    RunningInferenceProgressDialog(viewModel)
                    ModelInfoDialog(viewModel)
                }
            }
        }
    }

    @Composable
    private fun ColumnScope.SelectImagePanel(viewModel: MainActivityViewModel) {
        var selectedImage by remember { viewModel.selectedImageState }
        val pickMediaLauncher = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.PickVisualMedia()
        ) {
            if (it != null) {
                val bitmap = getFixedBitmap(it)
                selectedImage = bitmap
            }
        }
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.LightGray)
                .weight(1f),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (selectedImage == null) {
                Button(modifier = Modifier.padding(vertical = 40.dp), onClick = {
                    pickMediaLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                }) {
                    Icon(imageVector = Icons.Default.Add, contentDescription = "Select an image")
                    Text(text = "Select an image")
                }
            } else {
                Image(
                    bitmap = selectedImage!!.asImageBitmap(),
                    contentDescription = "Selected image",
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
    }

    @Composable
    private fun ColumnScope.EnterDescriptionPanel(viewModel: MainActivityViewModel) {
        var description by remember{ viewModel.descriptionState }
        val similarityScore by remember{ viewModel.similarityScoreState }
        Column(
            modifier = Modifier
                .fillMaxSize()
                .weight(1f)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                if (similarityScore == null) {
                    TextField(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 16.dp),
                        label = { Text(text = "Enter a description") },
                        value = description,
                        onValueChange = { description = it }
                    )
                    Button(
                        enabled = description.isNotEmpty(),
                        onClick = { viewModel.compare() }
                    ) {
                        Text(text = "Compare")
                    }
                }
                else {
                    Text(
                        text = "Similarity score: $similarityScore",
                        modifier = Modifier.padding(vertical = 16.dp),
                        fontSize = 24.sp
                    )
                    Button(onClick = { viewModel.reset() }) {
                        Text(text = "Compare again")
                    }
                }
            }
        }
    }

    @Composable
    private fun LoadModelProgressDialog(viewModel: MainActivityViewModel) {
        val isLoadingModel by remember{ viewModel.isLoadingModelState }
        if (isLoadingModel) {
            showProgressDialog()
            setProgressDialogText("Loading model...")
        }
        else {
            hideProgressDialog()
        }
        AppProgressDialog()
    }

    @Composable
    private fun RunningInferenceProgressDialog(viewModel: MainActivityViewModel) {
        val isInferenceRunning by remember{ viewModel.isInferenceRunning }
        if (isInferenceRunning) {
            showProgressDialog()
            setProgressDialogText("Running inference...")
        }
        else {
            hideProgressDialog()
        }
        AppProgressDialog()
    }

    @Composable
    private fun ModelInfoDialog(viewModel: MainActivityViewModel) {
        var showDialog by remember{ viewModel.isShowingModelInfoDialogState }
        if (showDialog && viewModel.visionHyperParameters != null && viewModel.textHyperParameters != null) {
            Dialog(
                onDismissRequest = { showDialog = false }
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp)
                        .background(Color.White, RoundedCornerShape(8.dp))
                        .padding(16.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Model Info",
                            modifier = Modifier.weight(1f),
                            style = MaterialTheme.typography.headlineLarge
                        )
                        IconButton(
                            onClick = { showDialog = false }
                        ) {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = "Close Model Info Dialog"
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(8.dp))

                    Text(text = "Vision Hyper-parameters", style = MaterialTheme.typography.bodyLarge)
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(text = "imageSize = ${viewModel.visionHyperParameters?.imageSize}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "hiddenSize = ${viewModel.visionHyperParameters?.hiddenSize}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "patchSize = ${viewModel.visionHyperParameters?.patchSize}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "projectionDim = ${viewModel.visionHyperParameters?.projectionDim}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "num layers = ${viewModel.visionHyperParameters?.nLayer}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "num intermediate = ${viewModel.visionHyperParameters?.nIntermediate}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "num heads = ${viewModel.visionHyperParameters?.nHead}", style = MaterialTheme.typography.labelSmall)
                    Spacer(modifier = Modifier.height(8.dp))

                    Text(text = "Text Hyper-parameters", style = MaterialTheme.typography.bodyLarge)
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(text = "num positions = ${viewModel.textHyperParameters?.numPositions}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "hiddenSize = ${viewModel.textHyperParameters?.hiddenSize}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "num vocab = ${viewModel.textHyperParameters?.nVocab}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "projectionDim = ${viewModel.textHyperParameters?.projectionDim}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "num layers = ${viewModel.textHyperParameters?.nLayer}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "num intermediate = ${viewModel.textHyperParameters?.nIntermediate}", style = MaterialTheme.typography.labelSmall)
                    Text(text = "num heads = ${viewModel.textHyperParameters?.nHead}", style = MaterialTheme.typography.labelSmall)
                }
            }
        }
    }

    private fun getFixedBitmap(imageFileUri: Uri): Bitmap {
        var imageBitmap = decodeStream(contentResolver.openInputStream(imageFileUri))
        val exifInterface = ExifInterface(contentResolver.openInputStream(imageFileUri)!!)
        imageBitmap = when (exifInterface.getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED
        )) {
            ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(imageBitmap, 90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(imageBitmap, 180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(imageBitmap, 270f)
            else -> imageBitmap
        }
        return imageBitmap
    }

    private fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, false)
    }

}

